rm(list=ls())

setwd("C:/Users/Abhishek/Desktop/Cab_Prac_DOS/R_cab")
getwd()

#install.packages("doSNOW")
#install.packages("rpart.plot")

#1. Load Libraries

x = c("ggplot2", "corrgram", "DMwR", "usdm", "caret", "randomForest", "e1071",
      "DataCombine", "doSNOW", "inTrees", "rpart.plot", "rpart",'MASS','xgboost','stats')

#load Packages
lapply(x, require, character.only = TRUE)

rm(x)



#2. Load dataset and study the data,details of attribute

train = read.csv("Train_cab.csv",header= T, na.strings = c("","","NA")) #Also replace blanks with "NA"
test= read.csv("test.csv",header=T)
test_pickup_datetime = test["pickup_datetime"]

str(train)
str(test)
# The details of data attributes in the dataset are as follows:
# pickup_datetime - timestamp value indicating when the cab ride started.
# pickup_longitude - float for longitude coordinate of where the cab ride started.
# pickup_latitude - float for latitude coordinate of where the cab ride started.
# dropoff_longitude - float for longitude coordinate of where the cab ride ended.
# dropoff_latitude - float for latitude coordinate of where the cab ride ended.
# passenger_count - an integer indicating the number of passengers in the cab ride.



# 3.Exploratory data Analysis
#3.1 Change datatype of required Variables
train$fare_amount= as.numeric(as.character(train$fare_amount))
train$passenger_count=round(train$passenger_count)

#3.2 Remove unrealistic values(Outliers) from Attributes¶

#fare Amount
train=train[-which(train$fare_amount<1),]

#passenger count
print(paste('Count for 6',nrow(train[which(train$passenger_count >6),]))) # to see counts > 6 [They are invalid]
print(paste('Count for 1',nrow(train[which(train$passenger_count <1),]))) # to see counts < 1 [They are invalid]
train=train[-which(train$passenger_count>6),]
train=train[-which(train$passenger_count<1),]

#pickup longitude,latitude
print(paste('pickup_longitude above 180=',nrow(train[which(train$pickup_longitude >180 ),])))
print(paste('pickup_longitude above -180=',nrow(train[which(train$pickup_longitude < -180 ),])))
print(paste('pickup_latitude above 90=',nrow(train[which(train$pickup_latitude > 90 ),]))) #ONLY THIS HAS OUTLIER
print(paste('pickup_latitude above -90=',nrow(train[which(train$pickup_latitude < -90 ),])))

#dropoff longitude,latitude
print(paste('dropoff_longitude above 180=',nrow(train[which(train$dropoff_longitude > 180 ),])))
print(paste('dropo ff_longitude above -180=',nrow(train[which(train$dropoff_longitude < -180 ),])))
print(paste('dropoff_latitude above -90=',nrow(train[which(train$dropoff_latitude < -90 ),])))
print(paste('dropoff_latitude above 90=',nrow(train[which(train$dropoff_latitude > 90 ),])))

# There's only one outlier which is in variable pickup_latitude.So we will remove it with nan.
# Also we will see if there are any values equal to 0.
nrow(train[which(train$pickup_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
nrow(train[which(train$dropoff_longitude == 0 ),])
nrow(train[which(train$dropoff_latitude == 0 ),])

# there are values which are equal to 0. we will remove them.
train = train[-which(train$pickup_latitude > 90),]

train = train[-which(train$pickup_longitude == 0),] 
train = train[-which(train$dropoff_longitude == 0),] 

##### Make a copy  This is like a checkpoint for the data After exploratory
df=train
#train=df





#4.Missing value Analysis
missing_val=data.frame(apply(train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val) #new column "Columns" has values as all rows in the index
names(missing_val)[1]="Missing_Percentage" #Rename 1st row name
missing_val$Missing_Percentage = (missing_val$Missing_Percentage/nrow(train)) * 100 #Convert data in missing percentage columns to actual percentages.
missing_val = missing_val[order(-missing_val$Missing_Percentage),] #Order percentage in descending
row.names(missing_val)=NULL
missing_val = missing_val[,c(2,1)] #interchange missing percentage and column position

#4.2 Impute the missing values
unique(train$passenger_count)
unique(test$passenger_count)  

train[,'passenger_count'] = factor(train[,'passenger_count'],labels = (1:6))
test[,'passenger_count'] = factor(test[,'passenger_count'],labels = (1:6))
train$passenger_count=factor(train$passenger_count,labels = (1:6))


train$passenger_count[3]=NA
train$passenger_count[3]

#train=mean(train$passenger_count,na.rm=T)
train = knnImputation(train, k = 181) #Removes all missing values

#Knn is giving closer results than mean,median. We did not consider Mode becuase of bias.
train$passenger_count[3]

sum(is.na(train)) #Now we have zero NA values
str(train)
summary(train)

df1=train   #Backup after missing value analysis
#train=df1



#5.Outlier Analysis
#only on numeric varisble
#stat_boxplot(geom = "errorbar", width=0.5) ADDD THIS FOR BETTER UNDERSTANDING

# Boxplot for fare_amount
pl1 = ggplot(train,aes(x = factor(passenger_count),y = fare_amount))
pl1 +stat_boxplot(geom = "errorbar", width=0.5) + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

# Replace all outliers with NA and impute
 #boxplot.stats HELPS TO DETECT AND REMOVE THE OUTLIERS, GIVR INSIDE IT THE NAME OF VARIABLE FOR WHICH YOU WANT OUTLIERS DETECTED 
 #$out FUNCTION WILL EXTRACT THE OUTLIERS
 #%in% SEARCH AND MATCH OPERATOR, WILL SEARCH ALL VALUES TO ITS RIGHT IN THE VARIABLE TO ITS LEFT. [MY RIGHT AND LEFT]
 #Vals will have all the outliers detected by the boxplot

vals = train[,"fare_amount"] %in% boxplot.stats(train[,"fare_amount"])$out #PUTS ALL OUTLIERS IN VALS VARIABLE
train[which(vals),"fare_amount"] = NA  #SET ALL OUTLIERS(Present in VALs) TO NA AND THEN IMPUTE

#lets check the NA's
sum(is.na(train$fare_amount))

#Imputing with KNN
train = knnImputation(train,k=3)

# lets check the missing values
sum(is.na(train$fare_amount))
str(train)

df2=train
#train=df2



## 6.Feature Engineering                       ##########################
# 1.Feature Engineering for timestamp variable
# we will derive new features from pickup_datetime variable
# new features will be year,month,day_of_week,hour
#Convert pickup_datetime from factor to date time

train$pickup_date = as.Date(as.character(train$pickup_datetime)) #	2009-06-15
train$pickup_weekday = as.factor(format(train$pickup_date,"%u")) # y=09(Year) ,u=1(weekday) ,m=06(month) ,d=15(day of month)
train$pickup_mnth = as.factor(format(train$pickup_date,"%m"))  
train$pickup_yr = as.factor(format(train$pickup_date,"%Y"))  #	2009	
pickup_time = strptime(train$pickup_datetime,"%Y-%m-%d %H:%M:%S")
train$pickup_hour = as.factor(format(pickup_time,"%H")) #17

#Add same features to test set

test$pickup_date = as.Date(as.character(test$pickup_datetime))
test$pickup_weekday = as.factor(format(test$pickup_date,"%u"))# Monday = 1
test$pickup_mnth = as.factor(format(test$pickup_date,"%m"))
test$pickup_yr = as.factor(format(test$pickup_date,"%Y"))
pickup_time = strptime(test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
test$pickup_hour = as.factor(format(pickup_time,"%H"))  

sum(is.na(train))# there was 1 'na' in pickup_datetime which created na's in above feature engineered variables.
train = na.omit(train) # we will remove that 1 row of na's  

sum(is.na(test))# there was 1 'na' in pickup_datetime which created na's in above feature engineered variables.
train = na.omit(train) # we will remove that 1 row of na's  


train = subset(train,select = -c(pickup_datetime,pickup_date)) #Removed this
test = subset(test,select = -c(pickup_datetime,pickup_date))  #removed this


# 2.Calculate the distance travelled using longitude and latitude
deg_to_rad = function(deg){
   (deg * pi) / 180
}
haversine = function(long1,lat1,long2,lat2){
   #long1rad = deg_to_rad(long1)
   phi1 = deg_to_rad(lat1)
   #long2rad = deg_to_rad(long2)
   phi2 = deg_to_rad(lat2)
   delphi = deg_to_rad(lat2 - lat1)
   dellamda = deg_to_rad(long2 - long1)
   
   a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
      sin(dellamda/2) * sin(dellamda/2)
   
   c = 2 * atan2(sqrt(a),sqrt(1-a))
   R = 6371e3
   R * c / 1000 #1000 is used to convert to meters
}

# Using haversine formula to calculate distance fr both train and test
train$dist = haversine(train$pickup_longitude,train$pickup_latitude,train$dropoff_longitude,train$dropoff_latitude)
test$dist = haversine(test$pickup_longitude,test$pickup_latitude,test$dropoff_longitude,test$dropoff_latitude)

# We will remove the variables which were used to feature engineer new variables
train = subset(train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
test = subset(test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))


train=train[-which(train$dist==0),]
test=test[-which(test$dist==0),]

str(train)
str(test)
summary(train)



## 7.Feature Selection                ###################
#1.corrgram : Correlation plot for numeric data , 2. Chi squarre : for factor data

numeric_index = sapply(train,is.numeric)
numeric_data = train[,numeric_index]

cnames = colnames(numeric_data)
cnames

corrgram(train[,numeric_index],upper.panel=panel.pie, main = "Correlation Plot")

#"fare_amount ~" is target variable, rest all all columns that are factors
aov_results = aov(fare_amount ~ passenger_count + pickup_hour + pickup_weekday + pickup_mnth + pickup_yr,data = train)

summary(aov_results)

#we saw in aov_result summary that weekday is Not significant so we remove.

#remove from train and test set
train = subset(train,select=-pickup_weekday)
test = subset(test,select=-pickup_weekday)



##################################             Feature Scaling         ################################################

#Normality check
qqnorm(train$fare_amount) #How to read
histogram(train$fare_amount)

   library(car)
# dev.off()
par(mfrow=c(1,2)) #What does it do
qqPlot(train$fare_amount)                       # qqPlot, it has a x values derived from gaussian distribution, if data is distributed normally then the sorted data points should lie very close to the solid reference line 
truehist(train$fare_amount)                     # truehist() scales the counts to give an estimate of the probability density.
lines(density(train$fare_amount))  # Right skewed      # lines() and density() functions to overlay a density plot on histogram

#Normalisation
#HERE THE DATA IS LEFT SKEWED THWREFORE WE USE NORMALIZATION
print('dist')
#Using the normalization formula here
train[,'dist'] = (train[,'dist'] - min(train[,'dist']))/
   (max(train[,'dist'] - min(train[,'dist'])))


## 9. Split the data into train and test data

set.seed(1000) #Random.seed is an integer vector, containing the random number generator (RNG) state for random number generation in R. It can be saved and restored, but should not be altered by the user.
library(caret)
library(Rcpp)
tr.idx = createDataPartition(train$fare_amount,p=0.75,list = FALSE) # 75% in trainin and 25% in Validation Datasets
train_data = train[tr.idx,] #new variable train_data which will have 75% of data
test_data = train[-tr.idx,] #new variable test_data which will have 25% of data


head(train_data)
summary(train_data)


#install.package("DataCombine") : DataCombine: Tools for Easily Combining and Cleaning Data Sets
library(DataCombine)
rmExcept(c("test","train","df",'df1','df2','df3','test_data','train_data','test_pickup_datetime'))

df22=train
df222=test

#train=df22
#test=df222


## 10. Select the model
#Error metric used to select model is RMSE

###########################################    1.Linear regression               #################

#LM IS FUNCTION TO HELP US BUILD LINEAR REGRESSION MODEL 
#TRAIN DATA TO BUILD THE MODEL
LR = lm(fare_amount ~.,data=train_data) #Fare_amount is the target variable

summary(LR) #more stra at pvalue = more significant variable
 str(train_data)


#PREDICT WILL HELP TO APPLY THE MODEL HERE
#TEST DATA TO APLLY THE MODEL
LR_predictions = predict(LR,test_data[,2:6])

#Mae(mean absolute error : Ans is no. of error), MAPE(Mean absolute percentage error : Ans is % of error),RMSE(Root mean squared error:Time based measure)
regr.eval(test_data[,1],LR_predictions) #Check how to read these outputs
# mae        mse       rmse       mape 
# 3.5303114 19.3079726  4.3940838  0.4510407  


#######################################      2. Decision Tree            #####################

DT = rpart(fare_amount ~ ., data = train_data, method = "anova") #Fare_amount is the target variable

summary(DT) #of the Nodes
#Predict for new test cases
DT_predictions = predict(DT, test_data[,2:6]) #2:6 because 1 is fare amount so we consider all columns except that

regr.eval(test_data[,1],DT_predictions)
# mae       mse      rmse      mape 
# 1.8981592 6.7034713 2.5891063 0.2241461 



#########################################################  3. Random forest            #####################
RF = randomForest(fare_amount ~.,data=train_data)

summary(RF)

RF_predictions = predict(RF,test_data[,2:6])


regr.eval(test_data[,1],RF_predictions)
# mae       mse      rmse      mape 
# 1.9053850 6.3682283 2.5235349 0.2335395

############          Improving Accuracy by using Ensemble technique ---- XGBOOST             ###########################
train_data_matrix = as.matrix(sapply(train_data[-1],as.numeric))
test_data_data_matrix = as.matrix(sapply(test_data[-1],as.numeric))

XGB = xgboost(data = train_data_matrix,label = train_data$fare_amount,nrounds = 15,verbose = FALSE)

summary(XGB)
XGB_predictions = predict(XGB,test_data_data_matrix)


regr.eval(test_data[,1],XGB_predictions)
# mae       mse      rmse      mape 
# 1.6183415 5.1096465 2.2604527 0.1861947  

#############                         Finalizing and Saving Model for later use                         ####################
# In this step we will train our model on whole training Dataset and save that model for later use
train_data_matrix2 = as.matrix(sapply(train[-1],as.numeric))
test_data_matrix2 = as.matrix(sapply(test,as.numeric))

XGB_MODEL = xgboost(data = train_data_matrix2,label = train$fare_amount,nrounds = 15,verbose = FALSE)

# Saving the trained model
saveRDS(XGB_MODEL, "./final_Xgboost_model_using_R.rds")

# loading the saved model
super_model <- readRDS("./final_Xgboost_model_using_R.rds")
print(super_model)

# Lets now predict on test dataset
xgb_fare = predict(super_model,test_data_matrix2)

#test["Predicted_Fare"]=xgb
test$Predicted_fare = xgb_fare
test

xgb_pred = test

# Now lets write(save) the predicted fare_amount in disk as .csv format 
write.csv(xgb_pred,"xgb_predictions_R.csv",row.names = FALSE)











   







