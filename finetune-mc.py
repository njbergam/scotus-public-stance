
import os
os.environ['NVIDIA_VISIBLE_DEVICES']="$gpu_id"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print('WHICH GPU ARE WE USING?', os.environ["CUDA_VISIBLE_DEVICES"])

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
python finetune.py --trn_data ../data/convote/train.csv --dev_data ../data/convote/dev.csv --num_classes 3 --model ../models/bert.pt --save_to ../models/convote.pt --simple True
python eval.py --test_data ../data/convote/test.csv --model ../models/convote.pt

(11)
python finetune.py --trn_data ../data/sc-stance/L2c/sc-stance-neu-ner-train.csv --dev_data ../data/sc-stance/L2c/sc-stance-neu-ner-dev.csv --num_classes 3 --model ../models/legal-bert.pt --save_to ../models/legal-sc-L2c.pt 
python eval.py --test_data ../data/sc-stance/L2c/sc-stance-neu-ner-test.csv --model ../models/legal-sc-L2c.pt 

(20)
python finetune.py --trn_data ../data/sc-stance/L2/sc-stance-neu-train.csv --dev_data ../data/sc-stance/L2/sc-stance-neu-dev.csv --num_classes 3 --model ../models/legal-bert.pt --save_to ../models/legal-sc-L2.pt 
python eval.py --test_data ../data/sc-stance/L2/sc-stance-neu-test.csv --model ../models/legal-sc-L2.pt 


"""


class StanceDatasetMC(Dataset):
    def __init__(self, df, tokenizer, device = torch.device("cuda")):#, tokenizer=BertTokenizerFast.from_pretrained('bert-base-uncased') ):
        #df['tok_post'] = df.apply(lambda row: word_tokenize(row['post']), axis=1)
        self.label = df['label'].values
        self.op1 = []
        self.op2 = []
        self.op3 = []
        self.op4 = []

        self.attn = []
        print('loading DataSet!')
        for i in tqdm(range(int(len(doc)))):
            s = ['[CLS] ' + statement.values[i] + ' [SEP] ' + doc.values[i] + ' [SEP]']
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
    pred_flat = np.argmax(preds, axis=1)#np.argmax(preds[0].cpu(), axis=1).flatten().detach().numpy()
    labels_flat = labels.cpu().flatten().detach().numpy()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def f1(preds, labels):
    y_pred = np.argmax(preds, axis=1)#.flatten().detach().numpy()
    y_true = labels.cpu().flatten().detach().numpy()
    return f1_score(y_true, y_pred, average='macro')


# function to train the model
def train(model):
    # define the optimizer
    optimizer = AdamW(model.parameters(),lr = 1e-5, weight_decay=1e-3)          # learning rate

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
        softmax = nn.LogSoftmax(dim=1)
        preds = softmax(model(sent_id, mask).logits) #model(sent_id, mask)#


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
            softmax = nn.LogSoftmax(dim=1)
            preds = softmax(model(sent_id, mask).logits) #model(sent_id, mask)#

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
    parser.add_argument('--legal_adapter', dest='legal_adapter',help='whether we use a legal adapter or not', required=False, default = False)

    args = parser.parse_args()

    


    tokenizer = None
    model = None
    if args.model.split('/')[-1].split('-')[0] == 'legal':
        print('Using legal BERT!')
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased", )
        model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=args.num_classes, output_hidden_states=True, attention_probs_dropout_prob=0.3,hidden_dropout_prob=0.3)
        model.load_state_dict(torch.load(args.model))
    elif args.model.split('/')[-1].split('-')[0] == 'long':
        print('Using longformer!')
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096") 
        model = AutoModelForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=args.num_classes, output_hidden_states=True, attention_probs_dropout_prob=0.3,hidden_dropout_prob=0.3)#BertModel(BertConfig()) #AutoModelForSequenceClassification.from_config(config)
        model.load_state_dict(torch.load(args.model))
    else:
        print('Using bert-base-uncased')
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') 
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args.num_classes, output_hidden_states=True, attention_probs_dropout_prob=0.3,hidden_dropout_prob=0.3)
        model.load_state_dict(torch.load(args.model))

    if args.legal_adapter:
        print('loading legal adapter...')
        # load pre-trained task adapter from Adapter Hub
        # this method call will also load a pre-trained classification head for the adapter task
        adapter_name = model.load_adapter('../models/adapters/legal@cu', config='pfeiffer')
        # activate the adapter we just loaded, so that it is used in every forward pass
        model.set_active_adapters(adapter_name)

    #model = BERT_Arch(model, num_labels=args.num_classes)

    df_train = pd.read_csv(args.trn_data)
    df_dev = pd.read_csv(args.dev_data)

    train_dls = []
    dev_dls = []
    for i in range(4):
        key = 'ox' + str(i)
        train_dls.append( DataLoader(StanceDataset(df_train['text'],df_train[key],df_train['label'], tokenizer), batch_size=32) )
        dev_dls.append( DataLoader(StanceDataset(df_train['text'],df_train[key],df_train['label'], tokenizer), batch_size=32) )


    # compute the class weights (based on an arbitrary subset)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_dls[0]['label']), y=train_dls[0]['label'])
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

    epochs = 20
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

        if len(valid_losses) > 3 and valid_losses[-1] > valid_losses[-2]:
            print('Validation loss increased. Implementing early stopping!')
            break

    print("Saving new model to ", str(args.save_to))
    torch.save(model.state_dict(), args.save_to)

