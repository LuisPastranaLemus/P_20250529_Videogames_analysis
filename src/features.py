import numpy as np
import pandas as pd

# Import function directly (more controlled than import *)
from src import *

def cast_datatypes(df):
    
    df['name'] = df['name'].astype('string')
    df = convert_object_to_numeric(df, type='float', include=['critic_score'])
    df = convert_object_to_numeric(df, type='float', include=['user_score'])
    df['platform'] = df['platform'].astype('category')
    df['genre'] = df['genre'].astype('category')
    df = normalize_datetime(df, include=['year_of_release'], frmt='%Y')
    
    return df
    