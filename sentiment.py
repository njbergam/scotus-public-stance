from unicodedata import category
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import requests
import re
import numpy as np
import scipy.stats as stats
import math
from sklearn.preprocessing import normalize


#df = pd.read_csv('opinions_to_html/all_opinions.csv')
#df.to_pickle('pickled_df.pkl')
df = pd.read_pickle("pickled_df.pkl")
df_sentiment = pd.read_csv('sentiment.csv')
df_year = pd.DataFrame({"year": [x[0:4] for x in df['date_filed']]})  
df = pd.concat([df,df_sentiment, df_year], axis=1)
#df = df.sort_values(by=['date_filed'], ascending=[True])


#print(df['neu'])

x = list(set([int(j[0:4]) for j in df['date_filed']]))
x.sort()
sentiment_by_year = []
agreement_by_year = []
for i in range(len(x)):
    #temp = []
    #for j in range(len(df)):
    #    if df['date_filed'][j][0:4] == x[i]:
    #        temp.append(df['neg'][j])
    k = [np.mean(df[df['year'] == str(x[i])]['neg']),np.mean(df[df['year'] == str(x[i])]['neu']),np.mean(df[df['year'] == str(x[i])]['pos']),np.mean(df[df['year'] == str(x[i])]['compound'])    ]
    sentiment_by_year.append(k)
    #print(k)
    k = [np.mean(df[df['year'] == str(x[i])]['scdb_votes_majority'])]
    #print(k)
    agreement_by_year.append(k[0])

print(x)

def normalize(l):
    l = np.array(l)
    mean = np.mean(l)
    sd = np.std(l)
    return [(x-mean)/sd for x in l]


print("\n\n ", agreement_by_year)

sent_types = ['neg', 'neu','pos','compound']
for i in range(len(sent_types)):
    sentiment = [j[i] for j in sentiment_by_year]
    #print(sentiment)
    #print(agreement_by_year)
    corr = stats.pearsonr(sentiment[0:-1], agreement_by_year[0:-1])
    print(sent_types[i], 'sentiment versus judge agreement over time')
    print("Corr:", corr)

    plt.plot(x[0:-1],normalize(sentiment[0:-1]), label=sent_types[i])
    plt.plot(x[0:-1],normalize(agreement_by_year[0:-1]), label = 'avg majority size')
    plt.title(corr)
    plt.legend()
    plt.savefig(sent_types[i] + '_versus_maj_size.png')
    plt.show()




def normalize(l):
    l = np.array(l)
    mean = np.mean(l)
    sd = np.std(l)
    return [(x-mean)/sd for x in l]

# x is a DICTIONARY OF LISTS of the time series variables
# special is the key of the value whose correlation we want to measure against the other variables
def plot_time_series(x, title):
    for k, v in x.items():
        # IMPORTANT: v is aligned 
        years = v[0]
        vals = v[1]
        plt.plot(years, normalize(vals), label = k)
        plt.title(title)
        plt.legend()
        plt.show()
        plt.savefig(title + '.png')
        







"""
df = pd.read_csv('synchrony_sample.csv')
overall_pearson_r = df.corr().iloc[0,1]
print(f"Pandas computed Pearson r: {overall_pearson_r}")

r, p = stats.pearsonr(df.dropna()['S1_Joy'], df.dropna()['S2_Joy'])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")

# Compute rolling window synchrony
f,ax=plt.subplots(figsize=(7,3))
df.rolling(window=30,center=True).median().plot(ax=ax)
ax.set(xlabel='Time',ylabel='Pearson r')
ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}")
"""
