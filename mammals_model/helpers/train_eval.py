import torch

import numpy as np

from torch import nn

from tqdm.notebook import tqdm

from helpers.metrics import MaskedAccuracy

from helpers.misc import EMA
    
def model_train(model, optimizer, dataloader, device, silent=False):

    criterion = torch.nn.CrossEntropyLoss(reduction = "mean")

    metric = MaskedAccuracy().to(device)
    
    model.train() #model to train mode

    if not silent:
        tot_itr = len(dataloader.dataset)//dataloader.batch_size #total train iterations
        pbar = tqdm(total = tot_itr, ncols=700) #progress bar

    loss_EMA = EMA()
    
    masked_acc, total_acc = 0., 0., 
        
    for itr_idx, ((masked_sequence, species_label), targets_masked, targets) in enumerate(dataloader):
        
        masked_sequence = masked_sequence.to(device)
        species_label = species_label.to(device)
        targets_masked = targets_masked.to(device)
        targets = targets.to(device)
            
        logits, _ = model(masked_sequence, species_label)

        loss = criterion(logits, targets_masked)

        optimizer.zero_grad()
        
        loss.backward()

        #if max_abs_grad:
        #    torch.nn.utils.clip_grad_value_(model.parameters(), max_abs_grad)

        optimizer.step()
                
        smoothed_loss = loss_EMA.update(loss.item())
            
        preds = torch.argmax(logits, dim=1)
        
        masked_acc += metric(preds, targets_masked).detach() # compute only on masked nucleotides
        total_acc += metric(preds, targets).detach()
        
        if not silent:

            pbar.update(1)
            pbar.set_description(f"acc: {total_acc/(itr_idx+1):.2}, masked acc: {masked_acc/(itr_idx+1):.2}, loss: {smoothed_loss:.4}")
         
    if not silent:
        del pbar
     
    return smoothed_loss, total_acc/(itr_idx+1), masked_acc/(itr_idx+1) 

def model_eval(model, optimizer, dataloader, device, save_embeddings = False, silent=False):

    criterion = torch.nn.CrossEntropyLoss(reduction = "mean")

    metric = MaskedAccuracy().to(device)
    
    model.eval() #model to train mode

    if not silent:
        tot_itr = len(dataloader.dataset)//dataloader.batch_size #total train iterations
        pbar = tqdm(total = tot_itr, ncols=700) #progress bar

    avg_loss, masked_acc, total_acc = 0., 0., 0.
    
    all_embeddings = []
        
    with torch.no_grad():

        for itr_idx, ((masked_sequence, species_label), targets_masked, targets) in enumerate(dataloader):

            masked_sequence = masked_sequence.to(device)
            species_label = species_label.to(device)
            targets_masked = targets_masked.to(device)
            targets = targets.to(device)

            logits, embeddings = model(masked_sequence, species_label)

            loss = criterion(logits, targets_masked)

            avg_loss += loss.item()
                
            preds = torch.argmax(logits, dim=1)

            masked_acc += metric(preds, targets_masked).detach() # compute only on masked nucleotides
            total_acc += metric(preds, targets).detach()
                
            if not silent:
                
                pbar.update(1)
                pbar.set_description(f"acc: {total_acc/(itr_idx+1):.2}, masked acc: {masked_acc/(itr_idx+1):.2}, loss: {avg_loss/(itr_idx+1):.4}")
                
            if save_embeddings:
                embeddings = embeddings['seq_embedding'].detach().cpu().numpy()
                all_embeddings.append(embeddings)

    if not silent:
        del pbar
     
    return (avg_loss/(itr_idx+1), total_acc/(itr_idx+1), masked_acc/(itr_idx+1)), all_embeddings
