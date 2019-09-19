import csv
import numpy as np 
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
import math
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_csv('/Users/rohanamarapurkar/Desktop/Exploratory-2.csv')
y=df['price']
df=df.drop('price',1)
df['host_response_rate'] = df['host_response_rate'].str.replace("%", "").astype("float")
df.drop(df.columns[0],axis=1,inplace=True)

def ConverttoFloat (df, string):
    
    ## Strings = security_deposit, cleaning_feem extra_people
    
    df[string]=df[string].str.replace('$', '')
    df[string]=df[string].str.replace(',', '')
    df[string] = df[string].astype('float64') 

    return df

def FeatureDrop (df):
    
    df=df.drop('calendar_last_scraped',1)
    df=df.drop('host_since',1)
    df=df.drop('host_verifications',1)
    df=df.drop('zipcode',1)
    df=df.drop('neighbourhood',1)
    df=df.drop('host_identity_verified',1)
    df=df.drop('require_guest_profile_picture',1)
    df=df.drop('require_guest_phone_verification',1)
    df=df.drop('id',1)
    return df


def CatToBin (df, string):
    
    ## String =  { property_type, neighbourhood_cleansed, cancellation_policy  }
    df = pd.get_dummies(df, columns = [string])
    return df

def CatToNum (df, data_clean):
    
    ## You need to define data_clean
    df.replace(data_clean, inplace=True)
    return df

def AmenitiesCount (df):
    
    count = df['amenities'].str.split(",").apply(len)
    df['amenities']=count
    return df

x  = FeatureDrop(df)
x = CatToBin(x, 'property_type')
x = CatToBin(x, 'neighbourhood_cleansed') 
x = CatToBin(x, 'cancellation_policy') 
x = CatToBin(x, 'host_is_superhost')
x = CatToBin(x, 'instant_bookable')
x = ConverttoFloat(x, 'security_deposit')
x = ConverttoFloat(x, 'cleaning_fee')
x = ConverttoFloat(x, 'extra_people')
x = AmenitiesCount(x)

cleanup_data = {"bed_type":     {"Real Bed": 5, "Futon": 4, "Pull-out Sofa":3, "Airbed":1,"Couch":2},
                "room_type": {"Entire home/apt": 3, "Private room": 2, "Shared room": 1}, 
                "host_response_time": {"a few days or more":4,"within a day":3,"within a few hours":2,"within an hour":1}
                }

x = CatToNum(x, cleanup_data)

x['host_response_time']=x['host_response_time'].fillna(x['host_response_time'].median())
x['host_response_rate']=x['host_response_rate'].fillna(x['host_response_rate'].mean())
x['review_scores_rating']=x['review_scores_rating'].fillna(x['review_scores_rating'].mean())
x['review_scores_checkin']=x['review_scores_checkin'].fillna(x['review_scores_checkin'].mean())
x['review_scores_cleanliness']=x['review_scores_cleanliness'].fillna(x['review_scores_cleanliness'].mean())
x['review_scores_accuracy']=x['review_scores_accuracy'].fillna(x['review_scores_accuracy'].mean())
x['review_scores_communication']=x['review_scores_communication'].fillna(x['review_scores_communication'].mean())
x['review_scores_location']=x['review_scores_location'].fillna(x['review_scores_location'].mean())
x['review_scores_value']=x['review_scores_value'].fillna(x['review_scores_value'].mean())
x['reviews_per_month']=x['reviews_per_month'].fillna(x['reviews_per_month'].mean())
x['bathrooms']=x['bathrooms'].fillna(x['bathrooms'].median())
x['beds']=x['beds'].fillna(x['beds'].median())
x['bedrooms']=x['bedrooms'].fillna(x['bedrooms'].median())
x['security_deposit']=x['security_deposit'].fillna(x['security_deposit'].mean())
x['cleaning_fee']=x['cleaning_fee'].fillna(x['cleaning_fee'].mean())

X = x.values
y = y.values
scaler=StandardScaler()

"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


## Cross Validation for a in range (len(alphas)):

scaler=StandardScaler()

n_alphas = 10
n_trials=50
alphas = np.logspace(-4, 2, n_alphas)
coefs = []
best_coefs=[]
R_scores_train=np.zeros((n_alphas,n_trials))
R_scores_test=np.zeros((n_alphas,n_trials))
mean_sq_train=np.zeros((n_alphas,n_trials))
mean_sq_test=np.zeros((n_alphas,n_trials))


for a in range (len(alphas)):
    for i in range(n_trials):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        x_train_unorm1 = scaler.fit(X_train)
        ## Normalised data
        x_train_norm1 = scaler.transform(X_train)
        x_test_norm1 = scaler.transform(X_test)
        lasso = linear_model.Lasso(alpha=alphas[a], fit_intercept=True)
        lasso.fit(x_train_norm1, y_train)
        y_pred_train=lasso.predict(x_train_norm1)
        y_pred_test=lasso.predict(x_test_norm1)
        coefs.append(lasso.coef_)
        R_scores_train[a][i]=(lasso.score(x_train_norm1,y_train))
        R_scores_test[a][i]=(lasso.score(x_test_norm1,y_test))
        mean_sq_train[a][i]=mean_squared_error(y_pred_train,y_train)
        mean_sq_test[a][i]=mean_squared_error(y_pred_test,y_test)


mean_accur_test=R_scores_test.mean(1)
mean_accur_train=R_scores_train.mean(1)
mean_mean_sq_train=mean_sq_train.mean(1)
mean_mean_sq_test=mean_sq_test.mean(1)

index_max_train=np.argmax(mean_accur_train)
index_max_test = np.argmax(mean_accur_test)
index_min_mean_sq_train=np.argmin(mean_mean_sq_train)
index_min_mean_sq_test=np.argmin(mean_mean_sq_test)

print('best R2 alpha:' ,alphas[index_max_test])
print('with mean R2 on test data :' ,mean_accur_test[index_max_test])
print('with mean  R2  on training data :', mean_accur_train[index_max_test])
print('lowest mean squared error alpha: ', alphas[index_min_mean_sq_test])
print('with mean squared error on test data :' ,mean_mean_sq_test[index_min_mean_sq_test])
print('with mean squared error  on training data :', mean_mean_sq_train[index_min_mean_sq_test])
print('best index for R2', index_max_test)
print('best index for Mean Sq Error', index_min_mean_sq_test)
print(alphas)

scaler=StandardScaler()
n_alphas = 10
n_trials=50
alphas = np.logspace(-4, 2, n_alphas)
coefs = []
best_coefs=[]
R_scores_train=np.zeros((n_alphas,n_trials))
R_scores_test=np.zeros((n_alphas,n_trials))
mean_sq_train=np.zeros((n_alphas,n_trials))
mean_sq_test=np.zeros((n_alphas,n_trials))
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
x_train_unorm1 = scaler.fit(X_train)
        ## Normalised data
x_train_norm1 = scaler.transform(X_train)
x_test_norm1 = scaler.transform(X_test)
lasso2 = linear_model.Lasso(alpha=2.15443469, fit_intercept=True)
lasso2.fit(x_train_norm1, y_train)
y_pred_train=lasso2.predict(x_train_norm1)
y_pred_test=lasso2.predict(x_test_norm1)
MSE_train=mean_squared_error(y_pred_train,y_train)
MSE_test=mean_squared_error(y_pred_test,y_test)

weights=lasso2.coef_
print(weights)

plt.plot(weights)
plt.ylabel('Weight value')
plt.xlabel('weights')
plt.show()

imp_weigh=np.where(weights>abs(25),weights,0)
print('important weights', imp_weigh)

nonzeroind = np.nonzero(weights)
print(nonzeroind)