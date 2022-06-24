from datasets import load_dataset
from datasets import Dataset
from transformers import AutoModelForSequenceClassification,TrainingArguments, Trainer
from transformers import DistilBertForSequenceClassification
import numpy as np
import pandas as pd
from datasets import load_metric
import torch
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import random
import time
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import sys
import argparse
from transformers import BertConfig
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers import LongformerTokenizer

"""
# CROSS!
python eval.py --test_data ../data/sc-stance/L1/sc-stance-test.csv --model ../models/sc.pt
python eval.py --test_data ../data/sc-stance/L1/sc-stance-test.csv --model ../models/legal-sc.pt
python eval.py --test_data ../data/sc-stance/L1/sc-stance-test.csv --model ../models/long-sc-L1.pt


python eval.py --test_data ../data/sc-stance/L2/sc-stance-neu-test.csv --model ../models/legal-sc-L2c.pt


python eval.py --test_data ../data/sc-stance/L2/sc-stance-neu-test.csv --model ../models/sc-L2.pt
python eval.py --test_data ../data/sc-stance/L2/sc-stance-neu-test.csv --model ../models/legal-sc-L2.pt

python eval.py --test_data ../data/sc-stance/L2c/sc-stance-neu-ner-test.csv --model ../models/sc-L2c.pt
python eval.py --test_data ../data/sc-stance/L2c/sc-stance-neu-ner-test.csv --model ../models/legal-sc-L2c.pt

python eval.py --test_data ../data/sc-stance/sc-stance.csv --model ../models/sc_pretrain.pt
python eval.py --test_data ../data/sc-stance/sc-stance-neu.csv --model ../models/sc_pretrain.pt
python eval.py --test_data ../data/sc-stance/sc-stance.csv --model ../models/sc_pretrain.pt
python eval.py --test_data ../data/sc-stance/sc-stance.csv --model ../models/sc_pretrain.pt

"""

class StanceDataset(Dataset):
    def __init__(self, doc, statement, label, tokenizer, device = torch.device("cuda")):#, tokenizer=BertTokenizerFast.from_pretrained('bert-base-uncased') ):
        #df['tok_post'] = df.apply(lambda row: word_tokenize(row['post']), axis=1)
        self.label = []
        for i in range(len(label)):
            if label[i] == -1.0:
                print('found!')
                self.label.append(2.0)
            else:
                self.label.append(label[i])
        self.doc = []
        self.attn = []
        print('loading DataSet!')
        for i in tqdm(range(len(doc))):
            s = '[CLS] ' + statement.values[i] + ' [SEP] ' + doc.values[i] + ' [SEP]'
            tokens = tokenizer(s,padding='max_length',truncation=True,return_attention_mask = True)
            self.attn.append(tokens['attention_mask'])
            self.doc.append(tokens['input_ids'])
            #print(new)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        #inp = torch.cat( (self.doc[idx] , self.topic[idx]),0 )
        return (torch.tensor(self.doc[idx]).to(device),torch.tensor(self.attn[idx]).to(device), torch.tensor(self.label[idx]).to(device))


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    #print(preds.shape)
    pred_flat = np.argmax(preds[0].cpu(), axis=1).flatten().detach().numpy()
    labels_flat = labels.cpu().flatten().detach().numpy()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def cross_entropy(preds, labels):
    loss = -np.sum(labels.float() * np.log(preds.float()))
    return loss/float(preds.shape[0])

# function for evaluating the model
def evaluate(val_dataloader, model):
    print("\nEvaluating...")
    # deactivate dropout layers
    model.eval()
    total_loss, total_accuracy = 0, 0
    t0 = time.time()
    # empty list to save the model predictions
    total_preds = []
    # iterate over batches
    for step,batch in enumerate(val_dataloader):
        # Progress update every 50 batches.
        if step % 10 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)     
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        labels = labels.to(device)
        # deactivate autograd
        with torch.no_grad(): 
            model = model.to(device)
            # model predictions
            preds = model(sent_id.to(device), mask.to(device))
            #preds = preds.to(device)
            # compute the validation loss between actual and predicted values

            #loss = cross_entropy(preds,labels)
            #total_loss = total_loss + loss.item()
            acc = flat_accuracy(preds, labels)
            total_accuracy += acc
            
            preds = np.argmax(preds[0].cpu(), axis=1).flatten().detach().numpy()#preds.cpu().detach().numpy()#.cpu().numpy()
            total_preds.append(preds)

    # compute the validation loss of the epoch
    #avg_loss = total_loss / len(val_dataloader) 
    avg_acc = total_accuracy / len(val_dataloader)
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)
    return avg_acc, total_preds #avg_loss, total_preds


if __name__ == '__main__':
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', dest='test_data', help='Name of the dev data file', required=True)
    parser.add_argument('--num_classes', dest='num_classes', help='Number of classes in the dataset',
                        required=False, default=3, type=int)
    parser.add_argument('--model', dest='model',help='if we are fine-tuning an existing model, where is it', required=False, 
        default = torch.load('../models/vanilla_3.pt'))
    parser.add_argument('--save_results', dest='save_to',help='Where to save results', required=False)
    args = parser.parse_args()

    df_test = pd.read_csv(args.test_data)
    """
        #config = BertConfig(num_labels=3)
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True, num_labels=3)

        #BertConfig.from_pretrained("bert-base-cased")
        model = AutoModelForSequenceClassification.from_config(config)#.from_pretrained('bert-base-cased')
        #model = torch.load(args.model)#BertModel()#AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
        model.load_state_dict(torch.load(args.model))
        model.eval()
    """
    tokenizer = None
    if args.model.split('/')[-1].split('-')[0] == 'legal':
        print('Using legal BERT!')
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=3)
        model.load_state_dict(torch.load(args.model))
        model.eval()
    if args.model.split('/')[-1].split('-')[0] == 'long':
        print('Using longformer!')
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096") 
        model = AutoModelForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=3)#BertModel(BertConfig()) #AutoModelForSequenceClassification.from_config(config)
        model.load_state_dict(torch.load(args.model))
        model.eval()
    else:
        print('Using bert-base-uncased')
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') 
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        model.load_state_dict(torch.load(args.model))
        model.eval()
    
    



    val_dataloader = DataLoader(StanceDataset(df_test['text'],df_test['target'],df_test['label'], tokenizer), batch_size=32)

    avg_acc,total_preds = evaluate(val_dataloader, model)

    #print(total_preds)
    #flat_preds = np.argmax(total_preds, axis=1).flatten().detach().numpy()
    #print("ACCURACY:", avg_acc)
    print(classification_report(df_test['label'], total_preds))


