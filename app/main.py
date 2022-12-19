from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from models.pytorch import PytorchDataset, PytorchMultiClass, get_device, train_classification, test_classification

#Read the lists of brewery names and beer_style to convert them into number/index
brew_name = pd.read_csv('./data/brewery_name.csv')   
beer_style = pd.read_csv('./data/beer_style.csv') 

pmc_bp = torch.load("./models/beer_prediction.pt")


#Inside the main.py file, instantiate a FastAPI() class and save it into a variable called app
app = FastAPI()

#Create a function called read_root() that will return a dictionary with Hello as key and World as value.
@app.get("/")
def read_root():
    return {
        "Project objective": "This web API based on Neural Network responds predicted a single/ multiple type(s) of beer by input parameter of appearance, aroma, palate or taste"
        "List of endpoints: "
        "Expected input parameters: "
        "Output format: "
        "Link to the Github repository"
    }


#Create a function called healthcheck() that will return a welcome message. Add a decorator to it in order to add a GET endpoint to app on /health with status code 200
@app.get('/health', status_code=200)
def healthcheck():
    return 'Neural Network is all ready to go!'

#Create a function called format_features() with brewery_name, review_aroma, review_appearance, review_palate,review_taste, and beer_abv as input parameters that will return a dictionary with the names of the features as keys and the inpot parameters as lists
def format_features(brew_index, review_aroma, review_appearance, review_palate, review_taste, beer_abv):
    return {
        'brewery_name': [brew_index],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste],
        'beer_abv': [beer_abv]
    }

"""
- input parameters: `brewery_name`, `review_aroma`, `review_appearance`, `review_palate`, and `review_taste`
- output: prediction as text
Add a decorator to it in order to add a GET endpoint to `app` on `/beer/type`
"""
@app.get("/beer/type")
def predict(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv):
    brew_index = list(brew_name['0'].squeeze())
    brew_index = brew_index.index(brewery_name)
    features = format_features(brew_index, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    obs = pd.DataFrame(features)
    obs_dataset = torch.Tensor(np.array(obs))
    with torch.no_grad():
         output = pmc_bp(obs_dataset) 
    output = torch.argmax(output).numpy().astype(int) 
    return {beer_style.squeeze()[output]}  


