# AIRLINE-FLIGHT-FARE-PREDICTION
This model predicts the fare of flight on the basis of old records . In the process of completing this project - data cleaning, data processing , balancing the data and converting the categorical data into numerical data.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Procedure](#procedure)
- [Installations](#installations)

- [Results](#results)
## Overview
In this case, I will be analyzing the flight fare prediction using Machine Learning dataset using essential exploratory data analysis techniques then will draw some predictions about the price of the flight based on some features such as what type of airline it is, what is the arrival time, what is the departure time, what is the duration of the flight, source, destination and more.

## Dataset
- Airline : So this column will have all the types of airlines like Indigo, Jet Airways, Air India, and many more.
- Date_of_Journey: This column will let us know about the date on which the passenger’s journey will start.
- Source: This column holds the name of the place from where the passenger’s journey will start.
- Destination: This column holds the name of the place to where passengers wanted to travel.
- Route: Here we can know about that what is the route through which passengers have opted to travel from his/her source to their destination.
- Arrival_Time: Arrival time is when the passenger will reach his/her destination.
- Duration: Duration is the whole period that a flight will take to complete its journey from source to destination.
- Total_Stops: This will let us know in how many places flights will stop there for the flight in the whole journey.
- Additional_Info: In this column, we will get information about food, kind of food, and other amenities.
- Price: Price of the flight for a complete journey including all the expenses before onboarding.
## Prerequisites
 jupyter notebook, python libraries: Numpy, Pandas, onehot encoding. label encoder , Matplotlib, NLP,machine learning.
## Procedure
- EDA: Learn the complete process of EDA
- Data analysis: Learn to withdraw some insights from the dataset both mathematically and visualize it.
- Data visualization: Visualising the data to get better insight from it.
- Feature engineering: We will also see what kind of stuff we can do in the feature engineering part.
## Installations
```r
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
```
## Results
I have done a complete EDA process, getting data insights, feature engineering, and data visualization as well so after all these steps I can go for the prediction using machine learning model-making steps.

