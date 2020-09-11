#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
import json
import time
import datetime

period1_datetime = datetime.date(2015,1,5)
period2_datetime = datetime.date(2015,5,5)


BASEURL = 'https://finance.yahoo.com/quote/'
name = 'LTC-USD'
# '/history?period1=1483225200&period2=1527458400&interval=1d&filter=history&frequency=1d'

download_url = 'https://query1.finance.yahoo.com/v7/finance/download/LTC-USD?'
crumbtest="RHIdwzBqOBp"

params_down = {
    'period1': 1483225200,
    'period2': 1527458400,
    'interval': '1d',
    'events': 'history',
    'crumb': '3LgxWcqtZLF'
}
params_base = {
        'period1': time.mktime(period1_datetime.timetuple()),
        'period2': time.mktime(period2_datetime.timetuple()),
        'interval': '1d',
        'filter': 'history',
        'frequency': '1d'
    }

HEADERS = {'Content': 'application/json'}

def main():    
    url = download_url + name 
    test_url=url+"?period1=1483225200&period2=1527458400&interval=1d&events=history&crumb=" + crumbtest
    #r = requests.get(download_url, headers=HEADERS, params=params_down)
    r = requests.get(test_url, headers=HEADERS, params=params_base)
    print(repr(r.status_code))
    
    response = r.json
    print(repr(response))
    with open('./test.txt', "w") as f:
        f.write(repr(response))
    
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




