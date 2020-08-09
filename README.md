# Project for Seminar "Multimodal Machine Learning"
##### authors: Sarah Hoppe, Yulia Grushetskaya

pyERA library from https://github.com/mpatacchiola/pyERA used as well as the materials of the article ***"Epigenetic Robotic Architecture"***

### Project structure
#### notebooks
contain jupyter notebooks  
***SOM.ipynb*** used for testing and visualizing results of pretrained som for Nao dataset  
to run it the pretrained som are required that are available at https://box.hu-berlin.de/d/271928edc87844cd8eeb/  
***SOM_other_data.ipynb*** used for training and testing of hub-SOM model for iCub dataset
#### database 
contains a test set for a Nao robot, as well as a training set and a test set used in work of M. Zambelli, A. Cully, and Y. Demiris ***"Multimodal representation models forprediction and control from partial information"*** and available here https://github.com/ImperialCollegeLondon/Zambelli2019_RAS_multimodal_VAE/tree/master/matlab/database  
training date set for the Nao robot is available here https://box.hu-berlin.de/d/271928edc87844cd8eeb/
#### Nao
contains world for Webbot simulation  and cotroller to collect data for the experiment described in report with Nao humanoid robot  
contains script for data preprocessing
#### Training
contains script for creating and traning of different SOM architectures
#### Testing
contains script with auxiliary functions for testing and visualizing SOMs and results of reconstructions of various modalities.

