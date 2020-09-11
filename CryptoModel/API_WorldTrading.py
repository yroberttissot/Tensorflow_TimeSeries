#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
import time
import datetime

#Website: https://www.worldtradingdata.com/home
#loggin to get API key
#does not work for currencies or crypto!

def get_historical_quotes_stocks(symbol, start_period, end_period):
    """Return a resquests.get with content in Json from worldtradingdata
    
    >>> get_historical_quotes_stocks(UHR.SW, str(datetime.date(2017,1,5)), str(datetime.date(2017,5,5))).json()
    
    {'name': 'UHR.SW',
     'history': {'2017-05-05': {'open': '403.50',
       'close': '412.40',
       ...
    """
    #different stocks can be found on https://www.worldtradingdata.com/download/list or https://www.worldtradingdata.com/search?q=swatch
    api_url = 'https://www.worldtradingdata.com/api/v1/history?symbol='
    api_token='ds1JrScjFpSWLjtgRS0iqZIFaOlGNThZ8hxd46N4SKQGuqVQ0wNko74W3JDP'
    params = {
        'date_from': period1_datetime,
        'date_to': period2_datetime
    }
    headers = {
        'Content': 'application/json',
        'Authorization': 'Bearer {0}'.format(api_token)
    }

    return requests.get(api_url+symbol, headers=headers, params=params)


symbol = 'UHR.SW' #can be multiple symbols separated with ,
period1_datetime = str(datetime.date(2017,1,5))
period2_datetime = str(datetime.date(2017,5,5))
r=get_historical_quotes_stocks(symbol, period1_datetime, period2_datetime)


# In[2]:


response = r.json
print(repr(response))
with open('./test.txt', "w") as f:
    f.write(repr(response))
r.json()


# In[3]:





# In[ ]:





# In[ ]:




