# Fraud Detection Data Science Project

This is a short data science challenge completed over a couple of days. The goal of the challenge is to build a basic system to detect whether customers are fraudulent, given their user data. 

The project is self contained within this repository, where the user can customise the input files and output paths in the config.json file. The repository also contains a Jupyter notebook documenting the process of exploring the data and developing of the code.

## Repository structure

The code has two .py scripts which can be run to complete the following tasks:

1. `train_model.py` conducts relevant data engineering and trains a random forest model on input data of the expected type, saving the trained model. 

Input: The input data on which the model will be trained should be a json file with the same format as the `customers.json` data within the `data/` folder of this repository. The filepath of this data should be given in the `config.json` file under the `training_file_path` key. By default, it is set to the the `customers.json` data within the `data/` folder of this repository.

Output: Running this script will output a trained Scikit-learn random forest model stored in `.joblib` format. The path and filename where this model file will be saved to can be customised in the `config.json` file under the `model_name` key. By default, it is set to save as 'model' within this directory. An example `model.joblib` file has also been included.


2. `run_model.py` applies a model created in `train_model.py` to any data of the expected type to make predictions on whether or not the users in that data are fraudulent, saving the results to a new file.

Input: The input data on which the model will make predictions should be a json file that can a) be the same format as the `customers.json` file or b) be the same format as the `customers.json` file, without the `fraudulent` key and corresponding values (so that predictions can be made of data where the fraudulent activity of the users is not known). The filepath of this data should be given in the `config.json` file under the `running_file_path` key. By default, it is set to the the `customers.json` data within the `data/` folder of this repository.

Output: Running this script will output a json file containing the emails of all users in the input file as keys and a label ('Fraudulent' or 'Not fraudulent'), indicating the prediction, as values. The path and filename where this json file will be saved to can be customised in the `config.json` file under the `output_filename` key. By default, it is set to save as `email_predictions.json` within the `output/` folder of this repository. An example `email_predictions.json` file has been included.

## How to run

Before running the code, the packages given in `requirements.txt` should be installed.

To run `train_model.py`, ensure that the `config.json` file contains the required input data path and model name (see the above 'repository structure' section for more details). Then, the following code can be run using the command line:

`python train_model.py`

To run `run_model.py`, ensure that the `config.json` file contains the required data path for prediction, model name for the previously trained model and output json filename. Then, the following code can be run using the command line:

`python run_model.py`