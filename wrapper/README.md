#This folder contains scripts of end-to-end screening process.

##VS_wrapper.py
Specific script used for virtual screening, when we have SMILE STRINGS for each
molecule and binary label(and potentially continuous label).
Workflow: Based on user information provided in VS_wrapper_config.json,
generates 2 different fingerprint(Morgan and MACCSkeys) for each molecule,
builds 2 models based on each fingerprint,
run stacking ensemble model(second layer model),
automatically select best model(either one of the first layer model or second
  layer model) based on cross-validation and test result,
use best model to predict the final test dataset. 
