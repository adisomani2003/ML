from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather Data
boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data
                    , columns=boston_dataset.feature_names)
features = data.drop(['INDUS','AGE'], axis = 1)
log_prices = np.log(boston_dataset.target)

target = pd.DataFrame(log_prices, columns = ['Price'])

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)
 

#property_stats = np.ndarray(shape=(1,11))
#We just created a n dimensional empty array

# property_stats[0][CRIME_IDX] = features['CRIM'].mean()
# property_stats[0][ZN_IDX] = features['ZN'].mean()
# property_stats[0][CHAS_IDX] = features['CHAS'].mean()

property_stats = features.mean().values.reshape(1,11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

def get_log_estimate(nr_rooms, 
                    students_per_classroom,
                    next_to_river = False,
                    high_confidence = True):
    #Configure Property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    #Optimizing for river location, we must observe that value can be either 0 or 1 so if-else makes sense
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    log_estimate = regr.predict(property_stats)[0][0]
    
    #Again high confidence-Yes and no
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    return log_estimate, upper_bound, lower_bound, interval
    

#Lets setup all we did in the last cell into a function

def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):
    
    #Adding details to docstring so that anyone who uses Shift+Tab can get more info
    """
    Estimate the price of property in Boston
    Keyword Arugument:
    rm -- number of rooms in the property
    ptratio -- number of students per teacher in the classroom for the school in the area
    chas -- True is the property is next to the Charles River, False otherwise
    large_range -- True for a 95% prediction interval, False for a 68% interval
    """
    
    
    if rm<1 or ptratio<1:
        print('That is unrealistic. Try again')
        return
    #What the return does is, as soon as if condition hits the code below is not run
    else:
    
    
        log_est, upper, lower, conf = get_log_estimate(rm, 
                                                       students_per_classroom=ptratio, 
                                                       next_to_river=chas, 
                                                       high_confidence=large_range)

        #Changing the scale
        dollar_est = np.e**log_est * 1000 * SCALE_FACTOR

        # Convert to today's dollars
        dollar_est = np.e**log_est * 1000 * SCALE_FACTOR
        dollar_hi = np.e**upper * 1000 * SCALE_FACTOR
        dollar_low = np.e**lower * 1000 * SCALE_FACTOR

        # Round the dollar values to nearest thousand
        rounded_est = np.around(dollar_est, -3)
        rounded_hi = np.around(dollar_hi, -3)
        rounded_low = np.around(dollar_low, -3) 

        print(f'The estimated property value is {rounded_est}.')
        print(f'At {conf}% confidence the valuation range is')
        print(f'USD {rounded_low} at the lower end to USD {rounded_hi} at the high end.')