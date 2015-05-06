import json,copy
import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from collections import defaultdict
d=defaultdict(list)
dic=dict()

# Converts Json data to manageable form

def convert(x):
    ob = json.loads(x)
    obcopy=copy.deepcopy(ob)
    for k, v in obcopy.items():
        if isinstance(v, list):
            ob[k] = ';'.join(str(v))
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob

# Converts JSON file to pandas Dataframe

def json2df(json_filename):
    df = pd.DataFrame([convert(line) for line in open(json_filename)])
    return df

# Concatenates all the reviews with same businessId into one list

def bid_tokens(data):
    for bid in data['business_id']:
        dataf=data.loc[data['business_id'] == bid]
        d[bid].append(dataf['text'])
    return d

# Classifies review as positive or negative using NaiveBayesAnalyzer

def classifier(word_features):
    for key,values in word_features.items():
        val=TextBlob(str(values),analyzer=NaiveBayesAnalyzer())
        dic[key]=val.sentiment.classification
    return dic

# Json to a pandas dataframe  Review dataset

data= json2df("yelp_academic_dataset_review.json")

# Json to pandas dataframe Business dataset

bdata=json2df("yelp_academic_dataset_business.json")

# to select only required columns from Business dataset

bdata=bdata[['business_id','city','latitude','longitude','name','review_count','stars','state','type']]

# Concatinates all the reviews with same businessId into one list

word_features=bid_tokens(data)

#Classfies review as positive or negative using NaiveBayesAnalyzer

cdata=classifier(word_features)

# Appends new column 'CLASS' to business dataset and adds classification Postive or Negative

for key in cdata.keys():
    bdata['CLASS'] = np.where(bdata['business_id']==key, cdata[key], 'NaN')

#Converts Pandas dataframe to a CSV file

print('please wait...')
bdata.to_csv('new.csv',sep=';', encoding='utf-8', index=False)
print('done')