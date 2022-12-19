from fastapi import FastAPI, Query, Response
from typing import List, Optional
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
#brew_name = pd.read_csv('../api/app/data/brewery_name.csv')   
#beer_style = pd.read_csv('../api/app/data/beer_style.csv') 
brew_name = pd.read_csv('./data/brewery_name.csv')   
beer_style = pd.read_csv('./data/beer_style.csv') 

#pmc_bp = torch.load("../api/app/models/beer_prediction.pt")
pmc_bp = torch.load("./models/beer_prediction.pt")
brewery_index = list(brew_name['0'].squeeze())

#Inside the main.py file, instantiate a FastAPI() class and save it into a variable called app
app = FastAPI()

#Create a function called read_root() that will return a dictionary with Hello as key and World as value.
@app.get("/")
def read_root():
    return {
        "Project objective": "This web API based on Neural Network responds predicted a single/ multiple type(s) of beer by input parameter of appearance, aroma, palate or taste, beer_abv",
        "List of endpoints": "'/', '/health', '/beer/type', '/beers/type'",
        "Expected input parameters": "brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float",
        "Output format": "Beer Style: string",
        "Link to the Github repository": "https://github.com/hitoshi0531/adv-dsi-2022-at2-api"
    }


#Create a function called healthcheck() that will return a welcome message. Add a decorator to it in order to add a GET endpoint to app on /health with status code 200
@app.get('/health', status_code=200)
def healthcheck():
    return 'Neural Network is all ready to go!'

"""
Create a function called format_features() that formats features into a dictionary
- input: brewery_name, review_aroma, review_appearance, review_palate, review_taste, and beer_abv 
- output: a dictionary with the names of the features as keys and the inpot parameters as lists
"""
def format_features(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    brewery_id = brewery_index.index(brewery_name)
    return {
        'brewery_name': [brewery_id],
        'review_aroma': [review_aroma],
        'review_appearance': [review_appearance],
        'review_palate': [review_palate],
        'review_taste': [review_taste],
        'beer_abv': [beer_abv]
    }

"""
Create a function called predict_beer() that output prediction of beer by pytorch_multi_classification method
- input parameters: `brewery_name`, `review_aroma`, `review_appearance`, `review_palate`, review_taste and `beer_abv`
- output: predicted data in array
"""
def predict_beer(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    obs = pd.DataFrame(features)
    obs_dataset = torch.Tensor(np.array(obs))
    with torch.no_grad():
         output = pmc_bp(obs_dataset) 
    output = torch.argmax(output).numpy().astype(int) 
    return output

"""
Create a function called item_counts() that is used by 'single_beer_style' function or 'multi_beers_style' and it outputs the numbers of input items
- input parameters: dict
- output: input items numbers
"""
def item_counts(dict):
    counts = sum([1 if isinstance(dict[x], (str, int))
                 else len(dict[x]) 
                 for x in dict])
    return counts

"""
Create a function called beer_type() that outputs a single prediction
- input parameters: single input set [`brewery_name`, `review_aroma`, `review_appearance`, `review_palate`, review_taste and `beer_abv`]
- output: a single prediction of beer style converted by predicted data in array
- Add a decorator to it in order to add a GET endpoint to app on '/beer/type/'
"""
@app.get("/beer/type")
def single_beer_style(brewery_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float):
    result = predict_beer(brewery_name, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
    return {'Beer Style': beer_style.squeeze()[result]}

"""
Create a function called beers_type() that outputs multiple predictions
- input parameters: multiple input sets [`brewery_name`, `review_aroma`, `review_appearance`, `review_palate`, review_taste and `beer_abv`]
- output: multiple predictions of beer style converted by predicted data in array
- Add a decorator to it in order to add a GET endpoint to app on '/beers/type/'
"""
@app.get("/beers/type/")
def multi_beers_style(beer_input: Optional[List[str]] = Query(None)):
    query_items = { "beer_input" : beer_input }
    df_beer = pd.DataFrame(columns=['brewery_name', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'beer_abv', 'brewery_id', 'beer_predict', 'beer_style'])
    for i in range(item_counts(query_items)):
        beer_val = [value[i] for value in query_items.values()]
        beer_str = ' '.join(beer_val).split(',')        
        dict = {'brewery_name': beer_str[0],  
                'review_aroma': float(beer_str[1]), 
                'review_appearance': float(beer_str[2]), 
                'review_palate': float(beer_str[3]), 
                'review_taste': float(beer_str[4]), 
                'beer_abv': float(beer_str[5]),
                'brewery_id': brew_index.index(beer_str[0]),
        }
        result = predict_beer(dict['brewery_name'],dict['review_aroma'],dict['review_appearance'],
                              dict['review_palate'],dict['review_taste'],dict['beer_abv'])
        dict.update({
                "beer_predict": result, 
                "beer_style": beer_style.squeeze()[result]
                })        
        df_beer = df_beer.append(dict, ignore_index = True)        
    return df_beer['beer_style'].to_dict()