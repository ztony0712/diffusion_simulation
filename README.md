# Diffusion Simulation Based on LeapfrogAV
This is a readme

## Envirnoment
LED 

## Training

``` 
cd ~/diffusion_simulation/LED && python main_led_nba.py
```
## Test
1. Current version has the diffusion noise, directly run this script:
```
python run_nuplan_test.py
```

2. If you want to remove the diffusion noise, go to this directory:
```
cd ~/LeapfrogAV/LED/models
```
Change the layers.py content by using wo_layers.py, and change the model_led_initializer.py by using wo_model_led_initializer.py. Then, run the other script:
```
python run_nuplan_test_wo_diffuison.py
```
The model files are stored in the following path
```
~/LeapfrogAV/LED/results/led_augment/models_folder/models
```
model_0120.p is diffusion version and the model_0330.p is no-diffusion version.

3. To run the existing test log files, you can use nuboard and import them.
``` bash
# Start the nuboard
'''
    The Planner folder must be in the 
    direcotry that nuboard start
'''
cd ~/diffusion_simulation && nuboard
```
Then update the .nuboard file in the experiment folder you indicate.

## To do
Try to figure out the problem in gameformer encoder.
Construct visulization pipline based on nuplan devkit.