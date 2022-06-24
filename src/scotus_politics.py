# loop throiugh all oral arguments
# loop through all political statements
# record when and where a political statement is made by a justice

import pandas as pd
import os
from eval import StanceDataset
import itertools
from IPython import embed
from transformers import BertConfig
from transformers import AutoModelForSequenceClassification,TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertTokenizerFast
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
nltk.download('vader_lexicon')


class StanceDataset(Dataset):
    def __init__(self, doc, statement, label, device = torch.device("cuda")):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        #df['tok_post'] = df.apply(lambda row: word_tokenize(row['post']), axis=1)
        self.label = label
        self.text = []
        self.attn = []
        for i in range(len(doc)):
            #print(doc.values[i], statement.values[i])
            txt = '[CLS] ' + str(doc[i]) + ' [SEP] ' + str(statement[i]) + ' [SEP]'
            tokens = tokenizer(txt, padding='max_length',truncation = True, return_attention_mask = True)
            self.attn.append(tokens['attention_mask'])
            self.text.append(tokens['input_ids'])
            #print(new)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        #inp = torch.cat( (self.doc[idx] , self.topic[idx]),0 )
        #return (torch.tensor(self.text[idx]).to(device),torch.tensor(self.attn[idx]).to(device), torch.tensor(self.label[idx]).to(device))
        return (torch.tensor(self.text[idx]),torch.tensor(self.attn[idx]), torch.tensor(self.label[idx]))


# DATAFRAME
# 1: date of case
# 2: case id
# 3: justice speaking
# 4: time of justice speaking
# 4: statement made by justice
# 5: political statement 
# 6: pro/con stance on that statement {-1,1}

def check_stance(justice_statement, political_statements):
    pairs = itertools.product(justice_statement, political_statements) # cartesian product

    x = []
    for p in political_statements:
        x.append(stance(j,p,model))
    return x
    #return (0,1,0,-1,1,0,-1) for instance


# 1-dimensional stance
def stance_bert(doc, statements, model):
    model.eval()
    d = [doc for i in range(len(statements))]
    dl = DataLoader(StanceDataset(d, statements, [1 for i in range(len(d))]), batch_size = len(d))
    for step, batch in enumerate(dl):
        model.eval()
        with torch.no_grad(): 
            preds = model(batch[0], batch[1])
            pred_flat = np.argmax(preds[0].cpu(), axis=1).flatten().detach().numpy()
            return pred_flat

# frequency of political statements made in oral arguments
def graph_politics():
    return 0

# we want to see how much this correlates with MQ scores
def graph_politics_by_judge(judge_list):
    return 0


def main():
     # If there's a GPU available...
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
    model.load_state_dict(torch.load('../models/VAST_model.pt'))
    model.eval()


    poll = pd.read_csv('../data/statements/pew_lib.csv')['statement'].values #pd.read_csv('../data/statements/scotus_topics.csv')['topics'].values

    rootdir = '../data/oyez-new'
    sia = SentimentIntensityAnalyzer()

    """
    for file in os.listdir(rootdir):
        print(file)
        d = os.path.join(rootdir, file)
        if os.path.isdir(d) and file != '__pycache__':
            cols = ['id', 'year', 'justice', 'time-stamp', 'statement', 'pos_sentiment', 'neg_sentiment', 'neu_sentiment'] + ['stance' + str(i) for i in range(len(poll))]
            full_df = pd.DataFrame(columns = cols)
            # going through each subdirectory
            # print(d + '/read_scripts.json')
            df = pd.read_json(d + '/read_scripts.json', typ='series')
            for i in tqdm(20): # going through 20 random transcripts...   #(range(len(df))):
                transcript = pd.read_json(df.iloc[i], typ='series')
                
                for j in range(len(transcript)):
                    #print("TEST: ", len(transcript))
                    #print("TEST2:", len(list(df.keys())))
                    if list(transcript.iloc[j]): # if list is not empty
                        #print("TESTING")
                        some_id = list(df.keys())[i].split('-')[-1]
                        year = list(df.keys())[i].split('_')[2].split('_')[0]
                        justice_name = list(transcript.iloc[j])[0][0]#.values[0][0]
                        time_stamp = list(transcript.iloc[j])[0][1]
                        justice_text = list(transcript.iloc[j])[0][3]#.values[0][3]
                        sent = sia.polarity_scores(justice_text)

                        add = [some_id, year, justice_name, time_stamp, justice_text, sent['pos'], sent['neg'], sent['neu']] + list(stance_bert(justice_text, poll, model))
                        new_row = pd.DataFrame({cols[i]: add[i] for i in range(len(add))}, index=[0])
                        full_df = pd.concat([full_df, new_row])
                    
            # intermediate versions of the dataset, done by year
            full_df.to_csv('../data/' + str(file) + 'sample_sentiment.csv')
    """

    years = [i + 1990 for i in range(22)]
    for y in years:
        cols = ['id', 'year', 'justice', 'time-stamp', 'statement', 'pos_sentiment', 'neg_sentiment', 'neu_sentiment'] + ['stance' + str(i) for i in range(len(poll))]
        full_df = pd.DataFrame(columns = cols)
        # going through each subdirectory
        # print(d + '/read_scripts.json')
        df = pd.read_json('../data/oyez-new/' + str(y) + '/read_scripts.json', typ='series')
        for i in tqdm(range(20)): # going through 20 random transcripts...   #(range(len(df))):
            transcript = pd.read_json(df.iloc[i], typ='series')
            
            for j in range(len(transcript)):
                #print("TEST: ", len(transcript))
                #print("TEST2:", len(list(df.keys())))
                if list(transcript.iloc[j]): # if list is not empty
                    #print("TESTING")
                    some_id = list(df.keys())[i].split('-')[-1]
                    year = list(df.keys())[i].split('_')[2].split('_')[0]
                    justice_name = list(transcript.iloc[j])[0][0]#.values[0][0]
                    time_stamp = list(transcript.iloc[j])[0][1]
                    justice_text = list(transcript.iloc[j])[0][3]#.values[0][3]
                    sent = sia.polarity_scores(justice_text)

                    add = [some_id, year, justice_name, time_stamp, justice_text, sent['pos'], sent['neg'], sent['neu']] + list(stance_bert(justice_text, poll, model))
                    new_row = pd.DataFrame({cols[i]: add[i] for i in range(len(add))}, index=[0])
                    full_df = pd.concat([full_df, new_row])

        full_df.to_csv('../data/' + str(y) + 'sample_sentiment.csv')   
        full_df = full_df.iloc[0:0]
        print("look at thtis empty boi", df)





main()