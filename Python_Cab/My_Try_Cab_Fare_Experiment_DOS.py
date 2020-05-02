#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os #getting access to input files


# # 1. Set Environment and load packages

# In[2]:


os.chdir("C:/Users/Abhishek/Desktop/Cab_Prac_DOS/Python_Cab")


# In[3]:


#Importing required libraries
import pandas as pd # Importing pandas for performing EDA
import numpy as np  # Importing numpy for Linear Algebric operations
import matplotlib.pyplot as plt # Importing for Data Visualization
import seaborn as sns # Importing for Data Visualization
from collections import Counter 
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #splitting dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import GridSearchCV
import joblib

#%matplotlib inline


# # 2. Load dataset and study the data,details of attribute

# In[4]:


#Loading the data:
train  = pd.read_csv("train_cab.csv",na_values={"pickup_datetime":"43"})
test   = pd.read_csv("test.csv")


# In[5]:


train.head(3)


# In[6]:


test.head(3)


# In[7]:


train.dtypes


# In[8]:


test.dtypes


# In[9]:


train.describe()


# In[10]:


test.describe()


# In[11]:


train.shape


# In[12]:


test.shape


# # 3.Exploratory data Analysis

# ### 3.1 Change datatype of required Variables

# In[13]:


da=pd.DataFrame(train. dtypes)
da


# In[14]:


train['fare_amount']=pd.to_numeric(train['fare_amount'],errors="coerce") #Using errors=’coerce’. It will replace all non-numeric values with NaN.


# In[15]:


train['pickup_datetime'].isna().sum()


# In[16]:


#Drop this Value before changing datatype
train=train.dropna(subset=['pickup_datetime'])


# In[17]:


train['pickup_datetime']=pd.to_datetime(train['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')


# In[18]:


train.dtypes


# ### 3.2 Remove unrealistic values(Outliers) from Attributes

# PASSENGER COUNT

# In[19]:


#Delete all <1 and >6
train["passenger_count"].describe()


# In[20]:


train=train.drop(train[train["passenger_count"]>6].index,axis=0)
train=train.drop(train[train["passenger_count"]<1].index,axis=0)


# In[21]:


print(train['passenger_count'].isna().sum())


# In[22]:


train.shape


# FARE AMOUNT

# In[23]:


train['fare_amount'].describe()
#min is -3.000
#Delete all below 0


# In[24]:


train=train.drop(train[train["fare_amount"]<1].index,axis=0)


# In[25]:


train['fare_amount'].sort_values(ascending=False)


# In[26]:


#In Fare amount There is a huge difference between top 3 and other values so we will remove the rows having fare amounting more that 454 as considering them as outliers

train = train.drop(train[train["fare_amount"]> 454 ].index, axis=0)
train.shape


#  pickup lattitude and longitude :

# In[27]:


#Longitude Range : -180 to 180
#Lattitude: -90 to 90

# we need to drop the rows having  pickup lattitute and longitute out the range mentioned above

print("pickup_latitude <90 :",train[train['pickup_latitude']<-90].index)
print("pickup_latitude >90 :",train[train['pickup_latitude']>90].index)
#there is one >90, have to drop that


# In[28]:


train = train.drop(train[train['pickup_latitude']>90].index,axis=0)


# In[29]:


train[train['pickup_longitude']<-180]
train[train['pickup_longitude']>180]


# In[30]:


train[train['dropoff_latitude']<-90]
train[train['dropoff_latitude']>90]


# In[31]:


train[train['dropoff_longitude']<-180]
train[train['dropoff_longitude']>180]


# In[32]:


train.shape


# # 4.Missing value Analysis

# In[33]:


test.isnull().sum()


# In[34]:


missing_val = pd.DataFrame(train.isnull().sum())


# In[35]:


missing_val=missing_val.reset_index()
missing_val


# In[36]:


missing_val=missing_val.rename(columns={"index" : "Variable",0:"Missing Percentage"})


# In[37]:


missing_val


# In[38]:


missing_val['Missing Percentage']=(missing_val["Missing Percentage"]/ len(train))*100


# In[39]:


missing_val=missing_val.sort_values("Missing Percentage",ascending=False)


# In[40]:


missing_val


# In[41]:


df=train
#train=df


# ## 4.2 Impute the missing values

# Try Mean,median,KNN.
# We wont be using mode because the most frequent value will dominate the imputing values

# #fare Amount :Value at [10] = 5.3,set it t0 0 and impute using :
# - using mean :11.37
# - using median :8.5
# - using KNN :5.97894
# 
# Therefore we will use KNN to impute the missing values

# In[42]:


# Choosing a random values to replace it as NA
a=train['fare_amount'].loc[10]
print('fare_amount at loc-10:{}'.format(a))
# Replacing 10 with NA
train['fare_amount'].loc[10] = np.nan
print('Value after replacing with nan:{}'.format(train['fare_amount'].loc[10]))
# Impute with mean
print('Value if imputed with mean:{}'.format(train['fare_amount'].fillna(train['fare_amount'].mean()).loc[10]))
# Impute with median
print('Value if imputed with median:{}'.format(train['fare_amount'].fillna(train['fare_amount'].median()).loc[10]))


# In[43]:


#Dropping pickup_datetime because KNN cant impute timestamp variable
columns=['fare_amount', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'passenger_count']
Pickup_datetime=pd.DataFrame(train['pickup_datetime'])


# In[44]:


train['fare_amount'].loc[1] = np.nan


# In[45]:


train = pd.DataFrame(KNNImputer(n_neighbors=19).fit_transform(train.drop('pickup_datetime',axis=1)),columns=columns, index=train.index)


# In[46]:


train['fare_amount'].iloc[10]


# In[47]:


a=train['fare_amount'].loc[1]
print('fare_amount at loc-1000:{}'.format(a))


# In[48]:


train['passenger_count']=train['passenger_count'].round()


# In[49]:


train['passenger_count'].unique()


# In[50]:


train.isnull().sum() #All values have been imputed


# # 5.Outlier Analysis

# In[51]:


plt.figure(figsize=(20,5)) #size in terms of width and height
plt.xlim(0,100) #Range to be displayed on x axis
sns.boxplot(x=train['fare_amount'],data=train,orient='h')
plt.title('Boxplot of fare_amount')
plt.show()


# In[52]:


plt.figure(figsize=(20,10))
plt.xlim(0,100)
_ = sns.boxplot(x=train['fare_amount'],y=train['passenger_count'],data=train,orient='h')
plt.title('Boxplot of fare_amount w.r.t passenger_count')
plt.show()


# In[53]:


#Define function to calculate the outlier and replace it with NA
def rem_outlier(col):
    #Np.percentile() will give the 75&25 percentile
    q75, q25 = np.percentile(train[col], [75 ,25])
    print("Q 75 is :",q75,"Q 25 is :",q25)
    #Calculate IQR
    iqr = q75 - q25
    #Calculate inner and outer fence
    min_Val = q25 - (iqr*1.5)
    max_Val = q75 + (iqr*1.5)
    print("Minimum is :",min_Val,"Maximum is :",max_Val)
    #Replace with NA
    train.loc[train[col] < min_Val,col] = np.nan
    train.loc[train[col] > max_Val,col] = np.nan


# In[54]:


# for i in num_var:
rem_outlier('fare_amount')
#min value is in minus so lower i.e Q25 is wrong but we have already deleted values less than 0 so its okay
#     rem_outlier('pickup_longitude')
#     rem_outlier('pickup_latitude')
#     rem_outlier('dropoff_longitude')
#     rem_outlier('dropoff_latitude')


# In[55]:


train["fare_amount"].describe()


# In[56]:


train["fare_amount"].describe()


# In[57]:


pd.DataFrame(train.isnull().sum())


# In[58]:


#Imputing with missing values using KNN
train = pd.DataFrame(KNNImputer(n_neighbors=19).fit_transform(train), columns = train.columns, index=train.index)


# In[59]:


pd.DataFrame(train.isnull().sum())


# # 6.Feature Engineering

# ### Derive field like year, month, day of the week, etc from datetime

# In[60]:


# we will Join 2 Dataframes pickup_datetime and train
train=pd.merge(Pickup_datetime,train,right_index=True,left_index=True)


# In[61]:


test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"],format= "%Y-%m-%d %H:%M:%S UTC")


# In[62]:


#test_pickup_datetime=test["pickup_datetime"]


# In[63]:


### we will saperate the Pickup_datetime column into separate field like year, month, day of the week, etc

test['year'] = test['pickup_datetime'].dt.year
test['Month'] = test['pickup_datetime'].dt.month
test['Date'] = test['pickup_datetime'].dt.day
test['Day'] = test['pickup_datetime'].dt.dayofweek
test['Hour'] = test['pickup_datetime'].dt.hour
test['Minute'] = test['pickup_datetime'].dt.minute


# In[64]:


### we will saperate the Pickup_datetime column into separate field like year, month, day of the week, etc

train['year'] = train['pickup_datetime'].dt.year
train['Month'] = train['pickup_datetime'].dt.month
train['Date'] = train['pickup_datetime'].dt.day
train['Day'] = train['pickup_datetime'].dt.dayofweek
train['Hour'] = train['pickup_datetime'].dt.hour
train['Minute'] = train['pickup_datetime'].dt.minute


# ### Derive distance using Latitudes and longitudes

# In[65]:


#As we know that we have given pickup longitute and latitude values and same for drop. 
#Calculate the distance Using the haversine formula, create a new variable :distance.
from math import radians, cos, sin, asin, sqrt

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km
# 1min 


# In[66]:


train['distance'] = train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)
test['distance'] = test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']].apply(haversine,axis=1)


# In[67]:


Counter(train['distance'] >4)


# In[68]:


train.head()


# In[69]:


test.head()


# In[70]:


train.dtypes


# In[71]:


plt.figure(figsize=(20,5)) #size in terms of width and height
plt.xlim(0,100) #Range to be displayed on x axis
sns.boxplot(x=train['distance'],data=train,orient='h')
plt.title('Boxplot of distance')
# plt.savefig('bp of distance.png')
plt.show()


# In[72]:


Counter(train['distance'] == 0)


# In[73]:


Counter(test['distance'] == 0)


# In[74]:


train.shape


# In[75]:


test['distance'].isna().sum()


# In[76]:


train['distance'].describe()


# In[77]:


train.head(5)


# In[78]:


drop = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'Minute']
train = train.drop(drop, axis = 1)
test = test.drop(drop, axis = 1)


# In[79]:


test.head()


# # 7.Feature Selection

# In[80]:


train.dtypes
#We will do correlation analysis as we have numeric Data


# In[81]:


df_cor=train


# In[82]:


#sns.pairplot(train)


# In[83]:


#Set heightand width of the plot
f, ax=plt.subplots(figsize=(7,5)) #plt=matplot library

#Generate correlation matrix
corr=df_cor.corr()

#plot using Seaborn library
 #sns is seaborn library,helps to plot visualization
 #masks will create individual logs or correlation matrix, np.zeros_like function will create Square shaped bracket for the pane 
 #cmap=sns.diverging_palette : Helps to set the colours 
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool) , cmap=sns.diverging_palette(220,10,as_cmap=True),square=True,ax=ax)


#     Extreme Red = Highly positively correlated data.
#     Extreme Blue= Highly Negatively Correlated data.
#      -OBSERVATION :
#     -1.year and month are highly negatively correlated
#     -2.Day and hour are highly negatively correlated
# 
#     -So we can delete one from each so that we dont have redundant information(year and day deletd)

# In[84]:


train.shape


# In[85]:


train=train.drop(['Day'],axis=1)
test=test.drop(['Day'],axis=1)


# In[86]:


train.head()


# # 8.Feature Scaling

# In[87]:


#Normality check of training data is uniformly distributed or not (CHECKING CONTINOUS VARIABLES ONLY)
#FARE_AMOUNT,DISTANCE ARE CONTINOUS, REST ALL ARE CATEGORICAL

for i in ['fare_amount', 'distance']:
    print(i)
    sns.distplot(train[i],bins='auto',color='green')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()


# In[89]:


train['distance'].describe()


# In[90]:


train['fare_amount'].describe()


# In[91]:


data=["distance"]
data


# In[92]:


#We are using Normalization because the data is Highly skewed
#AFTER NORMALIZATION THE DATA RANGE IN DISTANCE WILL BE FROM 0 TO 1


for i in data:
    print(i)
    train[i]= (train[i] - min(train[i]) )  /  (max(train[i]) - min(train[i]))


# In[93]:


#Normality check for test data is uniformly distributed or not-

sns.distplot(test['distance'],bins='auto',color='green')
plt.title("Distribution for Variable "+i)
plt.ylabel("Density")
plt.show()


# # 9. Split the data into train and test data

# In[94]:


train.head(10)


# In[95]:


##train test split for further modelling
X_train, X_test, y_train, y_test = train_test_split( train.iloc[:, train.columns != 'fare_amount'], 
                         train.iloc[:, 0], test_size = 0.20, random_state = 1)


# In[96]:


X_train.head(5) #(12765,7) pass_count,year,month,date,day,hour,distance
X_test.head(1)  #(3192) pass_count,year,month,date,day,hour,distance
y_train.head(1) #(12765) fareamount
y_test.head(1)  #(3192) fareamount


# In[97]:


y_train.describe()


# In[98]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # 10. Select the model

# ### Linear Regression Model :

# In[99]:


# Building the model on  training dataset
LR = LinearRegression().fit(X_train , y_train)


# In[100]:


#predicting using the model on train data
pred_train_LR = LR.predict(X_train)


# In[101]:


#predicting using the model on test data
pred_test_LR = LR.predict(X_test)


# In[102]:


##calculating RMSE for train data
RMSE_train_LR= np.sqrt(mean_squared_error(y_train, pred_train_LR))

##calculating RMSE for test data
RMSE_test_LR = np.sqrt(mean_squared_error(y_test, pred_test_LR))


# In[103]:


#Root Mean Squared Error
print("RMSE For Training data = "+str(RMSE_train_LR))
print("RMSE For Test data = "+str(RMSE_test_LR))


# In[104]:


#calculate R^2 for train data
from sklearn.metrics import r2_score
r2_score(y_train, pred_train_LR) 


# In[105]:


r2_score(y_test, pred_test_LR) 


# In[106]:


X_test.head(1)


# import joblib
# joblib.dump(fit_LR,"C:/Users/Abhishek/Desktop/Cab_Prac_DOS/Python_Cab/LR_Model.pkl")

# # # Load the model from the file 
# fit_LR_from_joblib = joblib.load('LR_Model.pkl')

# fit_LR_from_joblib

# ### Decision tree Model : 

# In[107]:


DecisionTreeRegressor


# In[108]:


DT = DecisionTreeRegressor(max_depth = 2).fit(X_train,y_train)


# In[109]:


#predicting using the model on train data
pred_train_DT = DT.predict(X_train)

#predicting using the model on test data
pred_test_DT = DT.predict(X_test)


# In[110]:


##calculating RMSE for train data
RMSE_train_DT = np.sqrt(mean_squared_error(y_train, pred_train_DT))

##calculating RMSE for test data
RMSE_test_DT = np.sqrt(mean_squared_error(y_test, pred_test_DT))

#Root Mean Squared Error
print("RMSE For Training data = "+str(RMSE_train_DT))
print("RMSE For Test data = "+str(RMSE_test_DT))


# In[111]:


## R^2 calculation for train data
r2_score(y_train, pred_train_DT) 


# In[112]:


## R^2 calculation for test data
r2_score(y_test, pred_test_DT) 


# In[113]:


#Calculate RMSE for test
DT_rmse = np.sqrt(mean_squared_error(y_test,pred_test_DT))
print('RMSE = ',(DT_rmse))


# ### Random Forest Model :

# In[114]:


#from sklearn.ensemble import RandomForestRegressor


# In[115]:


RF = RandomForestRegressor(n_estimators = 200).fit(X_train,y_train)


# In[116]:


#predicting using the model on train data
pred_train_RF = RF.predict(X_train)
#predicting using the model on test data
pred_test_RF = RF.predict(X_test)


# In[117]:


##calculating RMSE for train data
RMSE_train_RF = np.sqrt(mean_squared_error(y_train, pred_train_RF))

##calculating RMSE for test data
RMSE_test_RF = np.sqrt(mean_squared_error(y_test, pred_test_RF))


# In[118]:


#Root Mean Squared Error

print("RMSE For Training data = "+str(RMSE_train_RF))
print("RMSE For Test data = "+str(RMSE_test_RF))


# In[119]:


## calculate R^2 for train data
r2_score(y_train, pred_train_RF) 


# In[120]:


#Best 72.9
#calculate R^2 for test data
r2_score(y_test, pred_test_RF) 


# # Ensemble technique ---- XGBOOST

# In[121]:


#from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import accuracy_score


# In[122]:


model = XGBRegressor(max_depth=3,
                      subsample=1,
                      n_estimators=200,
                      learning_rate=0.1,
                      min_child_weight=1,
                      random_state=5
                     )


# In[123]:


XGB = model.fit(X_train,y_train)


# In[124]:


#predicting using the model on train data
pred_train_gb = XGB.predict(X_train)

#predicting using the model on test data
pred_test_gb = XGB.predict(X_test)


# In[125]:


##calculating RMSE for train data
RMSE_train_gb = np.sqrt(mean_squared_error(y_train, pred_train_gb))
##calculating RMSE for test data
RMSE_test_RF = np.sqrt(mean_squared_error(y_test, pred_test_gb))


# In[126]:


##calculating RMSE 
print("Root Mean Squared Error For Training data = "+str(RMSE_train_gb))
print("Root Mean Squared Error For Test data = "+str(RMSE_test_RF))


# In[127]:


## R^2 calculation for test data
r2_score(y_train, pred_train_gb)


# In[128]:


r2_score(y_test, pred_test_gb) #now 81


# # Apply Finalized model on the Test Data
# ###  Ensemble technique ---- XGBOOST

# In[129]:


model = XGBRegressor(max_depth=3,
                      subsample=1,
                      n_estimators=200,
                      learning_rate=0.1,
                      min_child_weight=1,
                      random_state=5
                     )

XGB = model.fit(X_train,y_train)


# In[130]:


#Save the Model
joblib.dump(XGB,"C:/Users/Abhishek/Desktop/Cab_Prac_DOS/Python_Cab/XGB_Model.pkl")


# In[131]:


#Load the model from the file 
XGB_Model = joblib.load('XGB_Model.pkl')


# In[132]:


predictions_XGB_test = XGB_Model.predict(test)


# In[133]:


predictions_XGB_test


# In[134]:


test['Predicted_fare'] = predictions_XGB_test


# In[135]:


test.head(1)


# In[136]:


test.to_csv('xgb_predictions_Python.csv')

