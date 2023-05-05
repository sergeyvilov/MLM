#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import pickle
import os
import gc

import pysam

import torch
from torch.utils.data import DataLoader, Dataset

import argparse


# In[2]:


from encoding_utils import sequence_encoders

import helpers.train_eval as train_eval    #train and evaluation
import helpers.misc as misc                #miscellaneous functions
from helpers.metrics import MaskedAccuracy

from models.spec_dss import DSSResNet, DSSResNetEmb, SpecAdd

parser = argparse.ArgumentParser("main.py")

input_params.fasta = '/s/project/mll/sergey/effect_prediction/MLM/fasta/240_mammals/240_mammals.shuffled.fa'
input_params.species_list = '/s/project/mll/sergey/effect_prediction/MLM/fasta/240_mammals/240_species.txt'
input_params.tot_epochs = 50
input_params.output_dir = './test'
input_params.train = True
input_params.val_fraction = 0.1
input_params.train_splits = 4
input_params.save_at = []
input_params.validate_every = 1

parser.add_argument("--fasta_fa", help = "FASTA file", type = str, required = True)
parser.add_argument("--species_list", help = "species list for integer encoding", type = str, required = True)
parser.add_argument("--output_dir", help = "dir to save predictions and model/optimizer weights", type = str, required = True)
parser.add_argument("--model_weight", help = "initialization weight of the model", type = str, default = None, required = False)
parser.add_argument("--optimizer_weight", help = "initialization weight of the optimizer, use only to resume training", type = str, default = None, required = False)
parser.add_argument("--val_fraction", help = "fraction of validation dataset to use", type = float, default = 0.1, required = False)
parser.add_argument("--validate_every", help = "validate every N epochs", type = int, default = 1, required = False)
parser.add_argument("--train", help = "batch size", action='store_true', default = False, required = False)

parser.add_argument("--train_splits", help = "batch size", type = int, default = 32, required = False)

parser.add_argument("--tot_epochs", help = "total number of training epochs", type = int, default = 200, required = False)
parser.add_argument("--batch_size", help = "batch size", type = int, default = 32, required = False)
parser.add_argument("--learning_rate", help = "learning rate", type = float, default = 5e-4, required = False)
parser.add_argument("--weight_decay", help = "Adam weight decay", type = float, default = 0.125, required = False)
parser.add_argument("--save_at", help = "epochs to save model/optimizer weights, 1-based", nargs='+', type = int, default = [], required = False)

input_params = vars(parser.parse_args())

input_params = misc.dotdict(input_params)

input_params.validate_at = misc.list2range(input_params.validate_at)
input_params.save_at = misc.list2range(input_params.save_at)

for param_name in ['output_dir', '\\',
'train_dataset', 'val_dataset', 'test_dataset', '\\',
'n_languages', 'input_height', '\\',
'train_fraction', 'val_fraction', 'validate_every', '\\',
'tot_epochs', 'save_at', '\\',
'model_weight', 'optimizer_weight', '\\',
'model_name', '\\',
'batch_size', 'learning_rate', '\\',
'seed','warmup_epochs', 'reduce_lr_after_epoch', 'lr_decay_gamma', '\\',
'tuplemaxloss']:

    if param_name == '\\':
        print()
    else:
        print(f'{param_name.upper()}: {input_params[param_name]}')


class SeqDataset(Dataset):
    
    def __init__(self, fasta_fa, seq_df, transform):
        
        self.fasta = pysam.FastaFile(fasta_fa)
        
        self.seq_df = seq_df
        self.transform = transform
        
    def __len__(self):
        
        return len(self.seq_df)
    
    def __getitem__(self, idx):
        
        seq = self.fasta.fetch(seq_df.iloc[idx].seq_name).upper()
                
        species_label = seq_df.iloc[idx].species_label
                
        masked_sequence, target_labels_masked, target_labels, mask, _ = self.transform(seq, motifs = {})
        
        masked_sequence = (masked_sequence, species_label)
        
        return masked_sequence, target_labels_masked, target_labels
    
    def close(self):
        self.fasta.close()


# In[4]:


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('\nCUDA device: GPU\n')
else:
    device = torch.device('cpu')
    print('\nCUDA device: CPU\n')
    #raise Exception('CUDA is not found')


# In[5]:


gc.collect()
torch.cuda.empty_cache()


# In[27]:





# In[19]:


seq_df = pd.read_csv(input_params.fasta + '.fai', header=None, sep='\t', usecols=[0], names=['seq_name'])
seq_df['species_name'] = seq_df.seq_name.apply(lambda x:x.split(':')[1])

#seq_df['seq_len'] = seq_df.seq_name.apply(lambda x:int(x.split(':')[-1]))
#seq_df = seq_df[seq_df.seq_len>60]

species_encoding = pd.read_csv(input_params.species_list, header=None).squeeze().to_dict()
species_encoding = {species:idx for idx,species in species_encoding.items()}
species_encoding['Homo_sapiens'] = species_encoding['Pan_troglodytes']

seq_df['species_label'] = seq_df.species_name.map({species:idx for idx,species in species_encoding.items()})

#seq_df = seq_df.sample(frac = 1., random_state = 1) #DO NOT SHUFFLE, otherwise too slow


# In[8]:


seq_df = seq_df.iloc[:2000]


# In[22]:


seq_transform = sequence_encoders.SequenceDataEncoder(seq_len = 2000, total_len = 2000, 
                                                      mask_rate = 0.15, split_mask = True)


# In[23]:


if input_params.train:
    
    N_train = int(len(seq_df)*(1-input_params.val_fraction))       
    train_df, test_df = seq_df.iloc[:N_train], seq_df.iloc[N_train:]
                  
    train_fold = np.repeat(list(range(input_params.train_splits)),repeats = N_train // input_params.train_splits + 1 )
    train_df['train_fold'] = train_fold[:N_train]

    train_dataset = SeqDataset(input_params.fasta, train_df, transform = seq_transform)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = 512, num_workers = 16, collate_fn = None, shuffle = None)

else:
                  
    test_df = seq_df
                  
test_dataset = SeqDataset(input_params.fasta, test_df, transform = seq_transform)
test_dataloader = DataLoader(dataset = test_dataset, batch_size = 512, num_workers = 16, collate_fn = None, shuffle = None)


# In[24]:


species_encoder = SpecAdd(embed = True, encoder = 'label', d_model = 128)

model = DSSResNetEmb(d_input = 5, d_output = 5, d_model = 128, n_layers = 4, 
                     dropout = 0., embed_before = True, species_encoder = species_encoder)

model = model.to(device) 

model_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(model_params, lr = 1e-4, weight_decay = 5e-4)


# In[28]:


last_epoch = 0

if input_params.model_weight:

    if torch.cuda.is_available():
        #load on gpu
        model.load_state_dict(torch.load(input_params.model_weight))
        if input_params.optimizer_weight:
            optimizer.load_state_dict(torch.load(input_params.optimizer_weight))
    else:
        #load on cpu
        model.load_state_dict(torch.load(input_params.model_weight, map_location=torch.device('cpu')))
        if input_params.optimizer_weight:
            optimizer.load_state_dict(torch.load(input_params.optimizer_weight, map_location=torch.device('cpu')))

    last_epoch = int(input_params.model_weight.split('_')[-3]) #infer previous epoch from input_params.model_weight

predictions_dir = os.path.join(input_params.output_dir, 'predictions') #dir to save predictions
weights_dir = os.path.join(input_params.output_dir, 'weights') #dir to save model weights at save_at epochs

if input_params.save_at:
    os.makedirs(weights_dir, exist_ok = True)


# In[29]:


def metrics_to_str(metrics):
    loss, total_acc, masked_acc = metrics
    return f'loss: {loss:.4}, total acc: {total_acc:.3f}, masked acc: {masked_acc:.3f}'


# In[ ]:


from utils.misc import print    #print function that displays time

if input_params.train:

    for epoch in range(last_epoch+1, input_params.tot_epochs+1):

        print(f'EPOCH {epoch}: Training...')

        train_dataset.seq_df = train_df[train_df.train_fold == (epoch-1) % input_params.train_splits]
        print(f'using train samples: {list(train_dataset.seq_df.index[[0,-1]])}')

        train_metrics = train_eval.model_train(model, optimizer, train_dataloader, device,
                            silent = True)

        print(f'epoch {epoch} - train, {metrics_to_str(train_metrics)}')

        if epoch in input_params.save_at: #save model weights

            misc.save_model_weights(model, optimizer, weights_dir, epoch)

        if input_params.val_fraction>0 and ( epoch==input_params.tot_epochs or
                            (input_params.validate_every and epoch%input_params.validate_every==0)):

            print(f'EPOCH {epoch}: Validating...')

            val_metrics, _ =  train_eval.model_eval(model, optimizer, test_dataloader, device,
                    silent = True)

            print(f'epoch {epoch} - validation, {metrics_to_str(val_metrics)}')

else:

    print(f'EPOCH {last_epoch}: Test/Inference...')

    test_metrics, test_embeddings =  train_eval.model_eval(model, optimizer, test_dataloader, device, 
                                                          save_embeddings = True, silent = True)

    print(f'epoch {last_epoch} - test, {metrics_to_str(test_metrics)}')

    os.makedirs(predictions_dir, exist_ok = True)

    with open(predictions_dir + '/test_embeddings.pickle', 'wb') as f:
        pickle.dump(test_embeddings, f)

print()
print(f'peak GPU memory allocation: {round(torch.cuda.max_memory_allocated(device)/1024/1024)} Mb')
print('Done')

