import pandas as pd
import requests

# SCOTUS DATASET
#df = pd.read_csv("old/SCDB_2021_01_caseCentered_LegalProvision.csv")
#print(df.columns)
#print(df['issueArea'])



#df = pd.read_csv("anes/anes.csv")
#df.to_pickle('anes_pickle.pkl')
#df = pd.read_pickle("anes/anes_pickle.pkl")
#print(df.head)
#print(df['VCF9245'])


year = '2021'
id = '20-480'
URL = "https://www.oyez.org/cases/"+year+"/"+id
headers = {'User-Agent': 'Mozilla/5.0 (compatible; YandexAccessibilityBot/3.0; +http://yandex.com/bots)'}
page = requests.get(URL,headers=headers)
print(page.text)

print(df.iloc[0,:])

topic = []
for i in range(len(df)):
    year = str(df['date_filed'][i])[0:4]
    URL = "https://www.oyez.org/cases/"+year+"/"+id
    page = requests.get(URL)

    stance = []

print(page.text)



def get_docket_id(absolute_url):
    page = requests.get(absolute_url)
    text = page.text.replace('\n', '')
    docket_id =  re.search('\n\d{2}-\d{3}\n', text).group(0)
    return docket_id


def get_question(docket_id, year):
    print(year)
    print(docket_id)
    URL = "https://www.oyez.org/cases/"+str(year)+"/"+str(docket_id)
    print(URL)
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
        page = requests.get(URL, auth=('user', 'pass'),headers=headers)
        print(page.text)
    except:
        print('url request failed')
    
    
url = df['absolute_url'][1000]
year = df['date_filed'][1000][0:4]
#get_question(get_docket_id(url), year)
get_question(74,1959)



# BASIC FACTS
print("Columns:", df.columns)
print("Number of opinions:", len(df))
print("Number of justices:", len(list(set(df['author_name']))))

def get_sentiment(text):
    return sia.polarity_scores(text)

# SENTIMENT OVER TIME
def record_sentiment():
    sia = SentimentIntensityAnalyzer()
    # writes to a dataframe 


    x = [int(x[0:4]) for x in df['date_filed']]
    y = []
    for i in range(len(df['text'])):
        print(i)
        #y.append( sia.polarity_scores(df['text'][i]) )
        y.append(get_sentiment(df['text'][i]))
    d = pd.DataFrame(data=y)
    d.to_csv('sentiment.csv')  

    df_year = pd.DataFrame({"year": [x[0:4] for x in df['date_filed']]})  
    df = pd.concat([df,df_sentiment, df_year], axis=1)

    x = list(set([j[0:4] for j in df['date_filed']]))

    y = []
    for i in range(len(x)):
        #temp = []
        #for j in range(len(df)):
        #    if df['date_filed'][j][0:4] == x[i]:
        #        temp.append(df['neg'][j])
        k = [np.mean(df[df['year'] == x[i]]['neg']),np.mean(df[df['year'] == x[i]]['neu']),np.mean(df[df['year'] == x[i]]['pos']),np.mean(df[df['year'] == x[i]]['compound'])    ]
        y.append(k )



    x = [int(l) for l in x]
    for i in range(4):
        l = ['neg', 'neu', 'pos','compound']
        y_final = [k[i] for k in y]
        plt.scatter(x,y_final)
        plt.title(str(l[i]))
        plt.xlabel('year')
        plt.ylabel(str(l[i]))
        plt.savefig(str(l[i]) + '.png', dpi=400)
        plt.clf()

