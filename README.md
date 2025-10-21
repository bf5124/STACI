
STACI: Spatio-Temporal Aleatoric Conformal Inference

##################################################################################
Scripts: These will recreate STACI paper results
#################

Packages are located in the environment.yml file

We include data in the data folder with MSS_data and AOD_data. 

The main run files for the STACI method are:
	1)main_AOD.py: For AOD air pollution dataset 
	2)main_MSS.py: For synthetic Mean Sea Surface Dataset 
Results will be saved in the empty results folder attached.

Set desired arguments in the args.py file and adjust FFN latent model parameters in the model.py file.

BayesNN.py includes the STACI neural network approximation

svgd.py includes training code for SVGD

conformal.py contains conformal fitting and cross validation code

Once the environment is set up and desired args are set, run:

1) python main_AOD.py 
2) python main_MSS.py 