import pandas as pd
"""
df_stance = pd.Dataframe()

df_opinions = pd.read_pickle("pickled_df.pkl")

# dictionary which maps Supreme Court DataBase
text = {df_opinions['scdb_id'][i]:df_opinions['text'][i] for i in range(len(df))}

#df_stance['text'] = df_opinions['text']

import glob

open('output.csv', 'w').close()

# this collects the OPENSMILE features and puts them in output.csv (which is then read by hw3_classify.py)
for i in range(1960, 2015):
    df = pd.read_json('OyezScraper-master/'+str(i)+'/')


"""


#print(df.head)
#print(df.columns)
#print(df['https://api.oyez.org/cases/1968/1047'])
#print("ID", df['ID'].values)
#print("docket_num", df['docket_number'].values)
#print("additional_docket_numbers", df['additional_docket_numbers'].values)
#print("href", df['href'].values)
#print("year", df['year'].values)
#df_opinions = df_opinions[df_opinions['year_filed'] == 1968]
#print(df_opinions.columns)

#print("SCDB_ID", df_opinions['scdb_id'].values)


# Questions are always framed such that "Yes" corresponds to being in favor of the first party.
# ruling = 1 = "Yes" = first party. ruling = 0 = "No" = second party.
# category is crucial; if "dissenting," we need to switch the ruling!
def check_stance(ruling, category):
    if category != 'dissenting':
        return ruling
    elif category == "dissenting":
        if ruling == 1:
            return 0
        elif ruling == 0:
            return 1
        else:
            return ruling
    elif ruling == 2:
        return 2



df_opinions = pd.read_csv('../data/all_opinions.csv') #pd.read_pickle("pickled_df.pkl")
#df_opinions = df_opinions[df_opinions['category'] == 'majority']
df_id = pd.concat([  
    pd.read_csv('../data/SCDB/scdb_modern_case.csv'),
    pd.read_csv('../data/SCDB/scdb_old_case.csv', encoding = "ISO-8859-1")
    ])


A = set(df_opinions['scdb_id'].values)
B = set(df_id['caseId'].values)
ids = list(A.intersection(B))
print("We have this many IDs to work with:", len(ids))

l = []
k = 0

YEARS = [str(i) for i in range(1955, 2020)]
YEARS.extend(["1789-1850","1850-1900","1900-1940","1940-1955"])

for y in YEARS:
    print('year', y)
    df = pd.read_json('../data/oyez-new/'+str(y)+'/case_details.json', orient='columns').T

    # looping through the shared cases between the opinions and id subsets
    for j in range(len(ids)):
        if ids[j].split('-')[0] == y: # correct year...
        
            docket_id = df_id[df_id['caseId'] == ids[j]]['docketId']
            d = df_id[df_id['caseId'] == ids[j]]['docket'].values[0]
            if not df[df['docket_number'] == d].empty: # if oyez has info for this docket number

                doc = df_opinions[df_opinions['scdb_id'] == ids[j]]['text']
                question = df[df['docket_number'] == d]['question']
                conclusion = df[df['docket_number'] == d]['conclusion']

                cat = df_opinions[df_opinions['scdb_id'] == ids[j]]['category']
                direct = df_id[df_id['caseId'] == ids[j]]['partyWinning']
                #print('pre', direct.values[0])
                #print('final', check_stance(direct.values[0],cat.values[0]))

                #print('ADUIT')
                #print(len(doc))
                #print(len(question))
                #print(len(conclusion))
                #print(len(cat))
                #print(len(dir))

                #if not question.empty and not conclusion.empty and not doc.empty and not cat.empty and not dir.empty:
                l.append([docket_id.values[0], doc.values[0],question.values[0],conclusion.values[0] , check_stance(direct.values[0],cat.values[0]), cat.values[0] ])
            #else: 
            #    print('oopsies!')


d = pd.DataFrame(l,columns = ['docket_id', 'opinion_text', 'question','response_summary','stance', 'category'])
d.to_csv('scotus_stance_full.csv')

d=pd.read_csv('scotus_stance_full.csv')

#print(len(d[d['stance'] != -1]))



"""l = []
for i in range(len(df_opinions)):
    print(df_opinions['scdb_id'][i])
    print(df_id['caseId'][i])
    print('\n\n')
    if df_opinions.iloc[i]['scdb_id'] in df_id['caseId']:
        # [term, case within term, number of dockets consolidated]
        docket_id = df_id[  str(df_id['caseId']) == df_opinions.iloc[i]['scdb_id']  ]['docket_id'].split('-')
        print(docket_id)
        print('hi')
        #doc = df_opinions['text']
        #topic = 
        #stance = 
        #l.append( ,   )
"""

"""
vast = pd.read_csv('VAST/vast_train.csv')
print(vast.head)
print(vast.columns)
print(vast['topic_str'])

"""

#print("federal_cite_one", df_opinions['federal_cite_one'].values)

#print("year_filed", df_opinions['year_filed'].values)
#print("category", df_opinions['category'].values)
#print(df_opinions[df_opinions['scdb_id']=='1913-149'])

#print(df_opinions.iloc[1])


"""
import requests
import json
response_API = requests.get('https://www.courtlistener.com/api/bulk-data/citations/all.csv.gz')
print(response_API.status_code)

"""