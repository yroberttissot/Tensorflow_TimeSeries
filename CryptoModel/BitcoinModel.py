#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import string
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as pp
get_ipython().run_line_magic('matplotlib', 'inline')
from os import listdir
from os.path import isfile, join


# In[2]:


####---------------------------DATA MANIPULATION ---------------------------------####
#load the currencies files
path='data/'
CurrenciesFilesPath = ['data/'+f for f in listdir(path) if isfile(join(path, f))]
for file in CurrenciesFilesPath:
    print(file)


# In[4]:


#now the idea is to put all these data together in a single variable, matching them with the date variable

#make a list of the data? -> a list of dataframes?
data=[pd.read_csv(f) for f in CurrenciesFilesPath]

#Remove the useless columns
#data=[currency.drop(['Date','Open','High','Low','Adj Close','Volume'],1) for currency in data]
data=[currency.drop(['Open','High','Low','Adj Close','Volume'],1) for currency in data]


# In[5]:


#Remove duplicates
data=[table.drop_duplicates(subset='Date') for table in data]


# In[6]:


data[4].head()


# In[7]:


from functools import reduce
#Rename the colone Close to the name of the currency
temp=[f.split('/')[1].split('.')[0] for f in CurrenciesFilesPath]
for i, Table in enumerate(data):
    Table.columns=['Date',temp[i]]


# In[8]:


#Set the index with all possible dates from the CSV files
all_index=set()
all_dates=set()
for table in data:
    for ind in table.index.values:
        all_index.add(ind)
        all_dates.add(table['Date'][ind])
print(len(all_dates))
type(all_dates)


# In[9]:


#Build DF with all dates possible
dates_list=list(all_dates)
Data={'Date': dates_list}
temp=pd.DataFrame(Data)
temp.sort_values(by=['Date'], inplace=True, ascending=True)
temp=temp.reset_index(drop=True)
temp.head()


# In[10]:


#add that DF to all other loaded CSVs
data.insert(0,temp)

#using join (it will take the column 'Date' as common index) and put nan for absent data:
dfs = [df.set_index('Date') for df in data]
df_final=dfs[0].join(dfs[1:])
print(len(df_final))

df_final


# In[ ]:





# In[11]:


#sets the minimum mean absolute error to the mean of the column to predict (must always be bigger
#or my model really sucks..)
min_mae=df_final['BTC-USD'].mean()
best_model='none'


# In[12]:


#---------------------------MACHINE LEARNING ---------------------------------####
#Construct the variables
    #X goes from day 0 to N-1
    #Y (to predict) goes from day 1 to N
    #Select the column from the df_final as variable to predict (Y)
X=df_final.copy()
print(len(X))
X=X.dropna(axis=0)
print('Without nan, len of X= '+ str(len(X['BTC-USD'])))
Y=X['BTC-USD'].iloc[1:].copy()
X=X[:-1]
print('Y len = ' + str(len(Y)))
print('X len =' + str(len(X)))
Y.tail()

#Remove 15% percent of the data to test later on, but on cronological order.
number_to_cutoff=int(round(0.15*len(Y),0))
X_test_tail=X[len(X)-number_to_cutoff:]
Y_test_tail=Y[len(Y)-number_to_cutoff:]
X=X[:-number_to_cutoff]
Y=Y[:-number_to_cutoff]
X


# In[13]:


#Split the data (train and verification)
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.

#maybe it makes NO SENSE TO RANDOMISE TRAIN AND VAL SINCE WE WANT TO PREDICT FUTURE MOMENTS
#train_X, val_X, train_Y, val_Y = train_test_split(X, Y, random_state = 0)
number_to_cutoff=int(round(0.20*len(Y),0))
train_X=X[:-number_to_cutoff]
train_Y=Y[:-number_to_cutoff]
val_X=X[len(X)-number_to_cutoff:]
val_Y=Y[len(Y)-number_to_cutoff:]


# In[14]:


#Build the model
#-----> ML Model : Tree
from sklearn.tree import DecisionTreeRegressor
# Fit the model. Specify a number for random_state to ensure same results each run 
# and look what number of leaf gives best mae
def get_mae_tree(max_leaf_nodes, train_X, val_X, train_Y, val_Y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_Y)
    preds_val= model.predict(val_X)
    mae=mean_absolute_error(val_Y, preds_val)
    return(mae)

# Fit model (see to not underfit or overfit the model)
print("--------DecisionTreeRegressor--------")
for max_leaf_nodes in [5, 50, 500, 5000, 10000]:
    my_mae = get_mae_tree(max_leaf_nodes, train_X, val_X, train_Y, val_Y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    if min_mae > my_mae:
        min_mae=my_mae
        best_model='DecisionTreeRegressor with max_leaf_nodes: ' + str(max_leaf_nodes) +" and mae: "+ str(my_mae)


# In[15]:


#-----> ML Model : Random Forest
from sklearn.ensemble import RandomForestRegressor

def get_mae_forest(max_leaf_nodes, train_X, val_X, train_Y, val_Y, n_estimators=50):
    model = RandomForestRegressor(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_Y)
    preds_val= model.predict(val_X)
    mae=mean_absolute_error(val_Y, preds_val)
    return(mae)

# Fit model (see to not underfit or overfit the model)
print("--------ForestRegressor--------")
for max_leaf_nodes in [5, 50, 500, 5000, 10000]:
    for trees in [5,20,50,80]:
        my_mae = get_mae_forest(max_leaf_nodes, train_X, val_X, train_Y, val_Y, trees)
        print("Max leaf nodes: %d \t trees: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, trees, my_mae))
        if min_mae > my_mae:
            min_mae=my_mae
            best_model='RandomForest with max_leaf_nodes: ' + str(max_leaf_nodes)+ ", number of trees: (estimators) "+ str(trees) +" and mae: "+ str(my_mae)


# In[16]:


best_model


# In[17]:


#Run the best model:
model = RandomForestRegressor(n_estimators=80, max_leaf_nodes=50, random_state=0)
model.fit(train_X, train_Y)
preds_val= model.predict(X_test_tail)


# In[18]:


#make a verification if the prediction of the modification of the value going up or down is right
#meaning -> we take the normal data (val_Y), look if (N-1)-N is positive or negative
#-> do the same with predicted data (preds_val), and compare!


#MAKES NO SENSE SINCE THE DATA VAL_Y IS NOT CONTINUOUS IN TIME!
#use full data? 
def change(x,y):
    return (y-x)/x

report_y=[]
report_pred=[]
for i, value in enumerate(Y_test_tail):
    if i+1 < len(Y_test_tail):
        report_y.append(change(value, Y_test_tail[i+1]))
        report_pred.append(change(preds_val[i],preds_val[i+1]))


# In[19]:


#verify if the increase/decrease was well predicted 

plotdata=[]
corrects=0
for i, value in enumerate(report_y):
    if report_y[i]>0 and report_pred[i]>0:
        corrects=corrects+1
        plotdata.append(+1)
    elif report_y[i]<0 and report_pred[i]<0:
        corrects=corrects+1
        plotdata.append(+1)
    else:
        plotdata.append(-1)
        
print('The percentage of increase/decrease well predicted is: ' + str(corrects/len(report_y)))

for i, value in enumerate(preds_val):
    print(
            "Val: " + str(round(Y_test_tail[i],2)) + 
            " \t \t \t ; \t Pred: " + str(round(preds_val[i],2)) 
         )
    if i<len(report_pred):
        print(
            "Has increase of: " + str(round(report_y[i],2)) +
            "\t\t ; \t Predicted future increase of: " + str(round(report_pred[i],3))
        )


# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


#try to make a loop where you train the model for X data, predict X+1, train the model for X+1 data, predict X+2 and so on
#idea!
#or the 3 next days, and after one day passes it retrains and predict the next 3 days...
X=df_final.copy()
X=X.dropna(axis=0)
Y=X['BTC-USD'].iloc[1:].copy()
X=X[:-1]

#Remove 15% percent of the data.
number_to_cutoff=int(round(0.15*len(Y),0))
X_test_tail=X[len(X)-number_to_cutoff:]
Y_test_tail=Y[len(Y)-number_to_cutoff:]
X=X[:-number_to_cutoff]
Y=Y[:-number_to_cutoff]


#For each loop -> add one day to training data and predict the 3 next days. Save it in an overall variable to make 
#precision calculations
correct_inc=0
total_preds=0
model = RandomForestRegressor(n_estimators=80, max_leaf_nodes=50, random_state=0)
nbr_days_to_predict=1 #for 1 day, it has more than 50%!

for i in range(0, number_to_cutoff-3):
    model.fit(X[:-number_to_cutoff+i], Y[:-number_to_cutoff+i])
    start_pred=len(X)-number_to_cutoff+i
    preds_val= model.predict(X[start_pred:start_pred+nbr_days_to_predict])
        
    #print("The mean absolute error is: " + str(mean_absolute_error(preds_val, Y[start_pred:start_pred+3])))
    for j, pred in enumerate(preds_val):
        #calculate the increase/decrease of the BTC value between day N and N+1 (actual and predicted)
        act_inc=round(change(Y[start_pred-1+j], Y[start_pred+j]),3)
        if j==0:
                pred_inc=round(change(Y[start_pred-1+j], pred),3) 
        else:
                pred_inc=round(change(preds_val[j-1], preds_val[j]),3) 
        
        total_preds=total_preds+1
        if act_inc > 0 and pred_inc >0:
            correct_inc=correct_inc+1
        elif act_inc < 0 and pred_inc <0:
            correct_inc=correct_inc+1
            
print("The percentage of direction of value change predicted correctly is: " + str(round(correct_inc/total_preds,2)) )
        


# In[ ]:





# In[23]:


val_Y[:3]


# In[ ]:





# In[ ]:




