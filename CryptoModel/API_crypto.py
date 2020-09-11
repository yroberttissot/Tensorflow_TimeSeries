#!/usr/bin/env python
# coding: utf-8

# In[3]:


import time
import datetime
import requests
import json
import hmac
import hashlib


# In[10]:


def get_historical_crypto_data(crypt, start_period, end_period):
    api_url='https://api.bitfinex.com/v1/history/'
    
    params = {
        'date_from': period1_datetime,
        'date_to': period2_datetime
    }
    headers = {
        'Content': 'application/json',
        #'Authorization': 'Bearer {0}'.format(api_token)
    }
    return requests.get(api_url+crypt, headers=headers, params=params)
    
crypt = 'usd' #can be multiple symbols separated with ,
period1_datetime = str(datetime.date(2017,1,5))
period2_datetime = str(datetime.date(2017,5,5))
r=get_historical_crypto_data(crypt, period1_datetime, period2_datetime)
print(repr(r))


# In[ ]:





# In[ ]:




