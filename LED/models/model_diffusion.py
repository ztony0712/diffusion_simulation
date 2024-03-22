import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear

from models.layers import PositionalEncoding, ConcatSquashLinear, VectorMapEncoder, spatial_transformer

class st_encoder(nn.Module):
	def __init__(self):
		super().__init__()
		channel_in = 2
		channel_out = 32
		dim_kernel = 3
		self.dim_embedding_key = 256
		self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
		self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)

		self.relu = nn.ReLU()

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_normal_(self.spatial_conv.weight)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
		nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
		nn.init.zeros_(self.spatial_conv.bias)
		nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
		nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

	def forward(self, X):
		'''
		X: b, T, 2

		return: b, F
		'''
		X_t = torch.transpose(X, 1, 2)
		X_after_spatial = self.relu(self.spatial_conv(X_t))
		X_embed = torch.transpose(X_after_spatial, 1, 2)

		output_x, state_x = self.temporal_encoder(X_embed)
		state_x = state_x.squeeze(0)

		return state_x


class social_transformer(nn.Module):
	def __init__(self):
		super(social_transformer, self).__init__()
		self.encode_past = nn.Linear(60, 256, bias=False)
		# self.encode_past = nn.Linear(48, 256, bias=False)
		self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

	def forward(self, h, mask):
		'''
		h: batch_size, t, 2
		'''
		# print(h.shape)
		h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
		# print(h_feat.shape)
		# n_samples, 1, 64
		h_feat_ = self.transformer_encoder(h_feat, mask)
		h_feat = h_feat + h_feat_

		return h_feat


class TransformerDenoisingModel(Module):

	def __init__(self, context_dim=256, tf_layer=2, output_dim=2):
		super().__init__()
		self.encoder_context = social_transformer()
		self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=80)
		self.concat1 = ConcatSquashLinear(output_dim, 2*context_dim, context_dim+3)
		self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=2, dim_feedforward=2*context_dim)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
		self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
		self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
		self.linear = ConcatSquashLinear(context_dim//2, output_dim, context_dim+3)


	def forward(self, x, beta, context, mask):
		batch_size = x.size(0)
		beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask)
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)

		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
		ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
		
		x = self.concat1(ctx_emb, x)
		final_emb = x.permute(1,0,2)
		final_emb = self.pos_emb(final_emb)
		
		trans = self.transformer_encoder(final_emb).permute(1,0,2)
		trans = self.concat3(ctx_emb, trans)
		trans = self.concat4(ctx_emb, trans) 
		return self.linear(ctx_emb, trans)
	

	def generate_accelerate(self, x, beta, context, mask):
		batch_size = x.size(0)
		num_pred = x.shape[1]
		num_frames = x.shape[2]
		beta = beta.view(beta.size(0), 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask)
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)

		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
		# time_emb: [11, 1, 3]
		# context: [11, 1, 256]
		ctx_emb = torch.cat([time_emb, context], dim=-1).repeat(1, num_pred, 1).unsqueeze(2)
		# x: 11, 10, 20, 2
		# ctx_emb: 11, 10, 1, 259
		x = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, num_frames, 512)
		# x: 110, 20, 512
		final_emb = x.permute(1, 0, 2) #dim: torch.Size([20, 440, 512])
		final_emb = self.pos_emb(final_emb) #dim: torch.Size([20, 20, 440, 512])
		
		trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, num_pred, num_frames, 512)
		# trans: 11, 10, 20, 512
		trans = self.concat3.batch_generate(ctx_emb, trans)
		trans = self.concat4.batch_generate(ctx_emb, trans)
		return self.linear.batch_generate(ctx_emb, trans)

class TransformerDenoisingModel_SpatialEnc(Module):

	def __init__(self, context_dim=256, tf_layer=2, output_dim=2):
		super().__init__()

		self.encoder_context = social_transformer()
		self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=80)
		self.concat1 = ConcatSquashLinear(output_dim, 2*context_dim, context_dim+3+256)
		self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=2, dim_feedforward=2*context_dim)
		self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
		self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3+256)
		self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3+256)
		self.linear = ConcatSquashLinear(context_dim//2, output_dim, context_dim+3+256)

		#MDF
		self._lane_len = 50
		self._lane_feature = 7
		self._crosswalk_len = 30
		self._crosswalk_feature = 3
		self._route_len = 50
		self._route_feature = 3
		self.lane_encoder = VectorMapEncoder(self._lane_feature, self._lane_len) #?dim: 7, 50
		self.crosswalk_encoder = VectorMapEncoder(self._crosswalk_feature, self._crosswalk_len) #?dim: 30, 3<
		self.route_encoder = VectorMapEncoder(self._route_feature, self._route_len)
		self.spatial_encoder = spatial_transformer(256)


	def forward(self, data, x, beta, context, mask):
		batch_size = x.size(0)
		beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask)
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)

		# vector maps #MDF
		map_lanes = data['map_lanes'].cuda()
		map_crosswalks = data['map_crosswalks'].cuda()
		map_routes = data['route_lanes'].cuda()

		# map encoding #MDF
		encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes) # torch.Size([4, 15, 256])
		encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks) #dim: torch.Size([4, 15, 256])
		encoded_map_routes, routes_mask = self.route_encoder(map_routes)
		# map_info = torch.cat([encoded_map_lanes, encoded_map_crosswalks], dim=1) #torch.Size([4, 215, 256])
		# map_mask = torch.cat([lanes_mask, crosswalks_mask], dim=1)
		map_info = torch.cat([encoded_map_lanes, encoded_map_crosswalks, encoded_map_routes], dim=1)
		map_mask = torch.cat([lanes_mask, crosswalks_mask, routes_mask], dim=1)
		spatial_embed = self.spatial_encoder(map_info, map_mask) # torch.Size([4, 215, 256]) <- old, new -> ([4, 265, 256])

		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
		ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
		
		x = self.concat1(ctx_emb, x)
		final_emb = x.permute(1,0,2)
		final_emb = self.pos_emb(final_emb)
		
		trans = self.transformer_encoder(final_emb).permute(1,0,2)
		trans = self.concat3(ctx_emb, trans)
		trans = self.concat4(ctx_emb, trans) 
		return self.linear(ctx_emb, trans)
	

	def generate_accelerate(self, data, x, beta, context, mask):
		batch_size = x.size(0)
		num_pred = x.shape[1]
		num_frames = x.shape[2]
		beta = beta.view(beta.size(0), 1, 1)          # (B, 1, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		context = self.encoder_context(context, mask)
		# context = context.view(batch_size, 1, -1)   # (B, 1, F)

		# vector maps #MDF
		map_lanes = data['map_lanes'].cuda()
		map_crosswalks = data['map_crosswalks'].cuda()
		map_routes = data['route_lanes'].cuda()

		# map encoding #MDF
		encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes) # torch.Size([4, 15, 256])
		encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks) #dim: torch.Size([4, 15, 256])
		encoded_map_routes, routes_mask = self.route_encoder(map_routes)
		# map_info = torch.cat([encoded_map_lanes, encoded_map_crosswalks], dim=1) #torch.Size([4, 215, 256])
		# map_mask = torch.cat([lanes_mask, crosswalks_mask], dim=1)
		map_info = torch.cat([encoded_map_lanes, encoded_map_crosswalks, encoded_map_routes], dim=1)
		map_mask = torch.cat([lanes_mask, crosswalks_mask, routes_mask], dim=1)
		spatial_embed = self.spatial_encoder(map_info, map_mask) # torch.Size([4, 215, 256]) <- old, new -> ([B, 256])
		# extend to B, 1, 256
		spatial_embed = spatial_embed.unsqueeze(1)

		time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
		# time_emb: [11, 1, 3]
		# context: [11, 1, 256]
		ctx_emb = torch.cat([time_emb, context, spatial_embed], dim=-1).repeat(1, num_pred, 1).unsqueeze(2)
		# x: 11, 10, 20, 2
		# ctx_emb: 11, 10, 1, 259
		x = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, num_frames, 512)
		# x: 110, 20, 512
		final_emb = x.permute(1, 0, 2) #dim: torch.Size([20, 440, 512])
		final_emb = self.pos_emb(final_emb) #dim: torch.Size([20, 20, 440, 512])
		
		trans = self.transformer_encoder(final_emb).permute(1, 0, 2).contiguous().view(-1, num_pred, num_frames, 512)
		# trans: 11, 10, 20, 512
		trans = self.concat3.batch_generate(ctx_emb, trans)
		trans = self.concat4.batch_generate(ctx_emb, trans)
		return self.linear.batch_generate(ctx_emb, trans)
