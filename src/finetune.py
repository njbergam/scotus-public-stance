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
from sklearn.metrics import f1_score
import torch.nn as nn

from sklearn.utils.class_weight import compute_class_weight
from IPython import embed
# optimizer from hugging face transformers
from transformers import AdamW




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
        self.doc = []
        self.attn = []
        print('loading DataSet!')
        for i in tqdm(range(int(len(doc)/1000))):
            if label[i] == -1.0:
                print('found!')
                self.label.append(2.0)
            else:
                self.label.append(label[i])

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

class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  

      _, cls_hs = self.bert(sent_id, attention_mask=mask)
      
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      
      # apply softmax activation
      x = self.softmax(x)

      return x


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
    pred_flat = np.argmax(preds, axis=1)#np.argmax(preds[0].cpu(), axis=1).flatten().detach().numpy()
    labels_flat = labels.cpu().flatten().detach().numpy()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def f1(preds, labels):
    y_pred = np.argmax(preds, axis=1)#.flatten().detach().numpy()
    y_true = labels.cpu().flatten().detach().numpy()
    return f1_score(y_true, y_pred, average='macro')


"""
def cross_entropy(preds, labels):
    preds = np.argmax(preds.detach().cpu().numpy(), axis=1) #preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    loss =   -np.sum(labels * np.log(preds))#-np.dot( labels,  np.log(preds)   )
    return loss/float(preds.shape[0])
"""

# function to train the model
def train(model):
    # define the optimizer
    optimizer = AdamW(model.parameters(),lr = 1e-5, weight_decay=1e-5)          # learning rate

    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds=[]

    # iterate over batches
    for step,batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        #embed()

        model = model.to(device)
        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        #softmax = nn.LogSoftmax(dim=1)
        preds = model(sent_id, mask)#softmax(model.bert(sent_id, mask).logits)


        #preds = np.argmax(outputs.logits.cpu().detach().numpy(), axis=1).flatten()

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds.to(device), labels.type(torch.LongTensor).to(device))

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds


# function for evaluating the model
def evaluate(model):

    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    t0 = time.time()

    # iterate over batches
    for step,batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
                
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():
            
            model = model.to(device)
            # model predictions
            #preds = model(sent_id, mask)
            softmax = nn.LogSoftmax(dim=1)
            preds = softmax(model.bert(sent_id, mask).logits)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds.to(device),labels.type(torch.LongTensor).to(device))

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)
            
            print("F1:", f1(preds, labels))
            print("Acc:", flat_accuracy(preds, labels))

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


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
    parser.add_argument('--trn_data', dest='trn_data', help='Name of the train data file', required=True)
    parser.add_argument('--dev_data', dest='dev_data', help='Name of the dev data file', required=True)
    parser.add_argument('--num_classes', dest='num_classes', help='Number of classes in the dataset',
                        required=False, default=3, type=int)
    parser.add_argument('--model', dest='model',help='if we are fine-tuning an existing model, where is it', required=False, 
        default = torch.load('../models/bert.pt'))
    parser.add_argument('--save_to', dest='save_to',help='Where to save results', required=False)
    args = parser.parse_args()

    


    tokenizer = None
    model = None
    if args.model.split('/')[-1].split('-')[0] == 'legal':
        print('Using legal BERT!')
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased", )
        model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=3, output_hidden_states=True)
        model.load_state_dict(torch.load(args.model))
    elif args.model.split('/')[-1].split('-')[0] == 'long':
        print('Using longformer!')
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096") 
        model = AutoModelForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=3, output_hidden_states=True)#BertModel(BertConfig()) #AutoModelForSequenceClassification.from_config(config)
        model.load_state_dict(torch.load(args.model))
    else:
        print('Using bert-base-uncased')
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') 
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3, output_hidden_states=True)
        model.load_state_dict(torch.load(args.model))

    model = BERT_Arch(model)

    df_train = pd.read_csv(args.trn_data)
    df_dev = pd.read_csv(args.dev_data)

    train_dataloader = DataLoader(StanceDataset(df_train['text'],df_train['target'],df_train['label'], tokenizer), batch_size=8)
    val_dataloader = DataLoader(StanceDataset(df_dev['text'],df_dev['target'],df_dev['label'], tokenizer), batch_size=8)


    #compute the class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df_train['label']), y=df_train['label'])
    print("Class Weights:",class_weights)
    # converting list of class weights to a tensor
    weights= torch.tensor(class_weights,dtype=torch.float)
    # push to GPU
    weights = weights.to(device)
    # define the loss function
    cross_entropy  = nn.NLLLoss(weight=weights) 


    print('Data loaded!')

    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    valid_losses=[]

    epochs = 4
    #for each epoch
    for epoch in range(epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        
        #train model
        train_loss, _ = train(model)
        
        #evaluate model
        valid_loss, _ = evaluate(model)
        
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')


