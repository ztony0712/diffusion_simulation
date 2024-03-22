import math
import time
import numpy as np
import matplotlib.pyplot as plt
from shapely import Point, LineString
import sys

# Add the path to the Python path
path_to_add = "/home/nuplan/GameFormer-Planner"
sys.path.append(path_to_add)
from Planner.planner_utils import *
from Planner.observation import *
from GameFormer.predictor import GameFormer


from LED.models.model_led_initializer import LEDInitializer_SpatialEnc as InitializationModel
from .state_lattice_path_planner import LatticePlanner

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring


class Planner(AbstractPlanner):
    def __init__(self, model_path, device=None):
        self._max_path_length = MAX_LEN # [m]
        self._future_horizon = T # [s] 
        self._step_interval = DT # [s]
        self._target_speed = 13.0 # [m/s]
        self._N_points = int(T/DT)
        self._model_path = model_path

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self._device = device
    
    def name(self) -> str:
        return "GameFormer Planner"
    
    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        self._trajectory_planner = TrajectoryPlanner()
        self._path_planner = LatticePlanner(self._candidate_lane_edge_ids, self._max_path_length)

    def _initialize_model(self):
        # The parameters of the model should be the same as the one used in training
        # self._model = GameFormer(encoder_layers=3, decoder_levels=2)
        self.model_initializer = InitializationModel(t_h=10, d_h=6, t_f=5*8, d_f=2, k_pred=20)
        
        # Load trained model
        print(self._model_path)
        self.model_initializer.load_state_dict(torch.load(self._model_path, map_location=self._device)['model_initializer_dict'])
        self.model_initializer.to(self._device)
        self.model_initializer.eval()
        # self._model.load_state_dict(torch.load(self._model_path, map_location=self._device))
        # self._model.to(self._device)
        # self._model.eval()
        
    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]
    
    def _get_reference_path(self, ego_state, traffic_light_data, observation):
        # Get starting block
        starting_block = None
        min_target_speed = 3
        max_target_speed = 15
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf

        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break
            
        # In case the ego vehicle is not on the route, return None
        if closest_distance > 5:
            return None

        # Get reference path, handle exception
        try:
            ref_path = self._path_planner.plan(ego_state, starting_block, observation, traffic_light_data)
        except:
            ref_path = None

        if ref_path is None:
            return None

        # Annotate red light to occupancy
        occupancy = np.zeros(shape=(ref_path.shape[0], 1))
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                conn_path = lane_conn.baseline_path.discrete_path
                conn_path = np.array([[p.x, p.y] for p in conn_path])
                red_light_lane = transform_to_ego_frame(conn_path, ego_state)
                occupancy = annotate_occupancy(occupancy, ref_path, red_light_lane)

        # Annotate max speed along the reference path
        target_speed = starting_block.interior_edges[0].speed_limit_mps or self._target_speed
        target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
        max_speed = annotate_speed(ref_path, target_speed)

        # Finalize reference path
        ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1) # [x, y, theta, k, v_max, occupancy]
        if len(ref_path) < MAX_LEN * 10:
            ref_path = np.append(ref_path, np.repeat(ref_path[np.newaxis, -1], MAX_LEN*10-len(ref_path), axis=0), axis=0)
        
        return ref_path.astype(np.float32)

    def data_preprocess(self, data):
        """
        Input:
            pre_motion_3D: torch.Size([32, 11, 10, dim]), [batch_size, num_agent, past_frame, dimension]
            fut_motion_3D: torch.Size([32, 11, 20, dim])
            fut_motion_mask: torch.Size([32, 11, 20])
            pre_motion_mask: torch.Size([32, 11, 10])
            traj_scale: 1
            pred_mask: None
            seq: nuplan
        Output:
            batch_size0
            traj_mask: torch.Size([BS*agent_num, BS*agent_num])
            past_traj: torch.Size([BS*agent_num, obs_len, dim_past])
            fut_traj: torch.Size([BS*agent_num, pred_len, dim_future])
        """
        # for key, value in data.items() :
        #     print (key, value.shape)

        batch_size = 1
        agent_num = data['pre_motion_3D'].shape[0]
        obs_len = data['pre_motion_3D'].shape[1]
        # pred_len = data['fut_motion_3D'].shape[2]

        traj_mask = torch.zeros(batch_size*agent_num, batch_size*agent_num).cuda()
        for i in range(batch_size):
            traj_mask[i*agent_num:(i+1)*agent_num, i*agent_num:(i+1)*agent_num] = 1.

        past_traj = data['pre_motion_3D'].cuda().contiguous().view(-1, obs_len, data['pre_motion_3D'].shape[2])  #dim: torch.Size([44, 10, 6])
        # fut_traj = data['fut_motion_3D'].cuda().contiguous().view(-1, pred_len, data['fut_motion_3D'].shape[3])  #dim: torch.Size([44, 20, 3])
        return batch_size, traj_mask, past_traj

    def _get_prediction(self, features, past_traj, traj_mask):
        
        sample_prediction, mean_estimation, variance_estimation = self.model_initializer(features, past_traj, traj_mask)
        sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
        loc = mean_estimation[:, None] + sample_prediction

        return loc
    
    def _plan(self, ego_state, history, traffic_light_data):
        # Construct input features
        features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)
        batch_size, traj_mask, past_traj = self.data_preprocess(features)

        # Infer prediction model
        with torch.no_grad():
            plan = self._get_prediction(features, past_traj, traj_mask)
            plan = plan[0][0]
            plan = torch.cat((plan, torch.zeros(plan.size(0), 1).cuda()), dim=1)

     
        states = transform_predictions_to_states(plan.cpu(), history.ego_states, self._future_horizon, DT) 
        trajectory = InterpolatedTrajectory(states)

        return trajectory
    
    def compute_planner_trajectory(self, current_input: PlannerInput):
        s = time.time()
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state # what is the observation 
        
        trajectory = self._plan(ego_state, history, traffic_light_data)
        print(f'Iteration {iteration}: {time.time() - s:.3f} s')

        return trajectory
