# LeapfrogAV
This is a readme


## Envirnoment
LED 

## Training

``` 
cd /LeapfrogAV/LED && python main_led_nba.py
```
## Test
Current version has the diffusion noise, directly run this script:
```
python run_nuplan_test.py
```

If you want to remove the diffusion noise, go to this directory:
```
cd /home/nuplan/LeapfrogAV/LED/models
```
Change the layers.py content by using wo_layers.py, and change the model_led_initializer.py by using wo_model_led_initializer.py. Then, run the other script:
```
python run_nuplan_test_wo_diffuison.py
```
The model files are stored in the following path
```
/home/nuplan/LeapfrogAV/LED/results/led_augment/models_folder/models
```
model_0120.p is diffusion version and the model_0330.p is no-diffusion version.