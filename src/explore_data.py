import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from nltk.tokenize import sent_tokenize
from IPython import embed

# exploring the data

sc_stance = pd.read_csv('scotus_stance_full.csv')
print("Size of SC stance:",  len(sc_stance))
print("number of positive pairs:", len(sc_stance[sc_stance['stance'] == 1]))
print("number of negative pairs:", len(sc_stance[sc_stance['stance'] == 0]))
print("number of unknown pairs:",len(sc_stance[sc_stance['stance'] == 2]))

print(sc_stance[sc_stance['stance'] == -1])

print("average words per opinion:", sum([len(sc_stance.iloc[i]['opinion_text'].split(' ')) for i in range(len(sc_stance))])/len(sc_stance)  )
print("average words per question:", sum([len(str(sc_stance.iloc[i]['question']).split(' ')) for i in range(len(sc_stance))])/len(sc_stance)  )

# Now consider the first opinion.
# we want to analyze the flow of the sentences carefully
# we will do this by plotting sentiment over time along with 
from nltk.sentiment import SentimentIntensityAnalyzer
row1 = sc_stance.iloc[0]
sentences = sent_tokenize(row1['opinion_text'])
sia = SentimentIntensityAnalyzer()
neg_sen = [sia.polarity_scores(sen)['neg'] for sen in sentences]
print(neg_sen)
embed()
"""

vast_train = pd.read_csv('../data/VAST/vast_train.csv')
print("Size of VAST train:",  len(vast_train))

vast_test = pd.read_csv('../data/VAST/vast_test.csv')
print("Size of VAST test:",  len(vast_test))

vast_dev = pd.read_csv('../data/VAST/vast_dev.csv')
print("Size of VAST dev:",  len(vast_dev))

opinions = pd.read_csv('../data/all_opinions.csv')
print("Size of opinions dataset:",  len(opinions))
print(opinions.columns)

# WHEN "PUBLIC OPINION" COMES UP
for i in range(len(opinions)):
    if "public opinion" in opinions.iloc[i]['text']:
        print(opinions.iloc[i]['date_filed'], opinions.iloc[i]['text'])
        print('\n')

"""

"""
scdb_modern_case = pd.read_csv('../data/SCDB/scdb_modern_case.csv')
scdb_old_case = pd.read_csv('../data/SCDB/scdb_old_case.csv', encoding = "ISO-8859-1")

#scdb_modern_justice = pd.read_csv('../data/SCDB/scdb_modern_justice.csv', encoding = "ISO-8859-1")
#scdb_old_justice = pd.read_csv('../data/SCDB/scdb_old_justice.csv', encoding = "ISO-8859-1", low_memory=False)

print("Number of SCOTUS cases (SCDB):",  len(scdb_modern_case) + len(scdb_old_case))

def plot_cases_metadata_1():
    # CASES
    years = np.append(scdb_modern_case['term'].values, scdb_old_case['term'].values)
    plt.hist(years, len(list(set(years))), facecolor='blue')
    plt.title('number of cases per year (total = ' + str(len(scdb_modern_case) + len(scdb_old_case)) + ')')
    plt.savefig('../graphs/case_dates.png')
    plt.close()

    # ISSUES
    issue_dict = {1: "Criminal Procedure", 2: "Civil Rights", 3: "First Amendment", 4: "Due Process", 5: "Privacy", 6: "Attorneys", 7: "Unions",8: "Economic Activity",9: "Judicial Power",10: "Federalism",11: "Interstate Relations", 12: "Federal Taxation", 13: "Miscellaneous", 14: "Private Action"}
    print(scdb_modern_case['issueArea'].values)
    issues = np.append(scdb_modern_case['issueArea'].dropna().values, scdb_old_case['issueArea'].dropna().values)
    print(issues)
    freq = {issue_dict[int(i)]:issues.tolist().count(i) for i in set(issues)}
    bins = [issue_dict[i] for i in list(set(issues))]
    print(bins)
    print(freq)
    plt.bar(list(set(issues)), freq.values(), label = bins)
    plt.legend(loc=(1.04,0.1))#(loc='right')
    plt.title('frequency of issues tackled by scotus')
    plt.savefig('../graphs/case_issues.png')
    plt.close()



def plot_cases_metadata_2():
    # CASES
    years = [int(x[0:4]) for x in opinions['date_filed'].values]
    plt.hist(years, len(list(set(years))), facecolor='blue')
    plt.title('number of cases per year (total = ' + str(len(scdb_modern_case) + len(scdb_old_case)) + ')')
    plt.savefig('../graphs/case_dates_2.png')
    plt.close()
    
    # CATEGORY
    category = opinions['category'].values
    freq = {category.tolist().count(i) for i in list(set(category))}
    plt.bar(list(set(category)), freq)
    plt.savefig('../graphs/case_types.png')
    
plot_cases_metadata_2()
"""


#https://docs.google.com/document/d/1K3LtZ5Z6Zxh9Xuf5Pu0P4UuPXa_rCuE6b2_gL1yLej8/edit
casehold = pd.read_csv('../data/casehold/casehold.csv')
print("Size of CaseHOLD dataset:",  len(casehold))
print(casehold.columns)
print(casehold.head)

overruling = pd.read_csv('../data/casehold/overruling.csv')
print("Size of Overruling dataset:",  len(overruling))
print(overruling.columns)
print(overruling.head)

