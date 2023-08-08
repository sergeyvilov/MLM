import torch

import numpy as np

from torch import nn

from tqdm import tqdm

from helpers.metrics import MaskedAccuracy

from helpers.misc import EMA

from sklearn.metrics import accuracy_score

from torch.nn.functional import log_softmax

import re
    
def model_train(model, optimizer, dataloader, device, silent=False):

    criterion = torch.nn.CrossEntropyLoss(reduction = "mean")

    metric = MaskedAccuracy().to(device)
    
    model.train() #model to train mode

    if not silent:
        tot_itr = len(dataloader.dataset)//dataloader.batch_size #total train iterations
        pbar = tqdm(total = tot_itr, ncols=700) #progress bar

    loss_EMA = EMA()
    
    masked_acc, total_acc = 0., 0., 
        
    for itr_idx, ((masked_sequence, species_label), targets_masked, targets, _) in enumerate(dataloader):
        
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


def model_eval(model, optimizer, dataloader, device, get_embeddings = False, get_motif_acc=False, temperature=None, selected_motifs = None, silent=False):
    
    criterion = torch.nn.CrossEntropyLoss(reduction = "mean")

    metric = MaskedAccuracy().to(device)

    model.eval() #model to train mode

    if not silent:
        tot_itr = len(dataloader.dataset)//dataloader.batch_size #total train iterations
        pbar = tqdm(total = tot_itr, ncols=700) #progress bar

    avg_loss, masked_acc, total_acc = 0., 0., 0.
    
    all_embeddings = []
    
    motif_probas = []
            
    with torch.no_grad():

        for itr_idx, ((masked_sequence, species_label), targets_masked, targets, seq) in enumerate(dataloader):
            
            if get_embeddings or get_motif_acc:
                #batches are generated by transformation in the dataset,
                #so remove extra batch dimension added by dataloader
                masked_sequence, targets_masked, targets = masked_sequence[0], targets_masked[0], targets[0]
                species_label = species_label.tile((len(masked_sequence),))

            masked_sequence = masked_sequence.to(device)
            targets_masked = targets_masked.to(device)
            targets = targets.to(device)
            species_label = species_label.long().to(device)
            
            logits, embeddings = model(masked_sequence, species_label)
            
            if temperature:
                logits /= temperature

            loss = criterion(logits, targets_masked)

            avg_loss += loss.item()
                
            preds = torch.argmax(logits, dim=1)

            masked_acc += metric(preds, targets_masked).detach() # compute only on masked nucleotides
            total_acc += metric(preds, targets).detach()
                        
            if get_motif_acc:
                                
                targets_masked = targets_masked.T.flatten()
                logits = torch.permute(logits,(2,0,1)).reshape(-1,5).detach()
                    
                masked_targets = targets_masked[targets_masked!=-100].cpu()
                masked_logits = logits[targets_masked!=-100].cpu()
                
                sm = log_softmax(masked_logits, dim=1)
                
                #target_probas = torch.gather(torch.exp(sm),-1, masked_targets.unsqueeze(1)).squeeze().numpy() #probs only for true label
                
                target_probas = torch.exp(sm)[:,:4].numpy()#probs for all bases
                
                seq_name = dataloader.dataset.seq_df.iloc[itr_idx].seq_name.split(':')[0]
                
                motif_probas.append((seq_name,target_probas))
                
                ##assert len(target_probas) == len(seq[0])
                #for motif in selected_motifs:
                #    for match in re.finditer(motif, seq[0]):
                #        avg_target_probas = target_probas[match.start():match.end()].mean()
                #        motif_probas.append((itr_idx, motif,match.start(),avg_target_probas))
                
                ##for motif_start, motif_end in motif_ranges:
                ##    motif_start, motif_end = motif_start.numpy(), motif_end.numpy() 
                ##    avg_target_probas = target_probas[int(motif_start):int(motif_end)].mean()
                ##    motif_probas.append((itr_idx, motif_start, motif_end,avg_target_probas))

            if get_embeddings:
                # only get embeddings of the masked nucleotide
                sequence_embedding = embeddings["seq_embedding"]
                sequence_embedding = sequence_embedding.transpose(-1,-2)[targets_masked!=-100]
                # shape # B, L, dim  to L,dim, left with only masked nucleotide embeddings
                # average over sequence 
                #print(sequence_embedding.shape)
                sequence_embedding = sequence_embedding.mean(dim=0) # if we mask
                #sequence_embedding = sequence_embedding[0].mean(dim=-1) # no mask

                sequence_embedding = sequence_embedding.detach().cpu().numpy()
                all_embeddings.append(sequence_embedding)
                
            if not silent:
                
                pbar.update(1)
                pbar.set_description(f"acc: {total_acc/(itr_idx+1):.2}, masked acc: {masked_acc/(itr_idx+1):.2}, loss: {avg_loss/(itr_idx+1):.4}")

    if not silent:
        del pbar
     
    return (avg_loss/(itr_idx+1), total_acc/(itr_idx+1), masked_acc/(itr_idx+1)), all_embeddings, motif_probas