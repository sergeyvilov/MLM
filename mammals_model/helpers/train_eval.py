import torch
from torch import nn

from tqdm.notebook import tqdm


def model_train(model, optimizer, dataloader, device, scheduler=None, use_tuplemax=True, max_abs_grad =None, silent=False):

    model.train() #model to train mode

    #torch.autograd.detect_anomaly(True)

    if use_tuplemax:
        criterion = losses.TuplemaxLoss()
    else:
        criterion = nn.CrossEntropyLoss() #binary cross-entropy

    if not silent:
        tot_itr = len(dataloader.dataset)//dataloader.batch_size #total train iterations
        pbar = tqdm(total = tot_itr, ncols=700) #progress bar

    beta = 0.98 #beta of running average, don't change

    avg_loss = 0. #average loss

    all_predictions = []

    softmax = torch.nn.Softmax(dim=1)

    for itr_idx, (tensors, labels) in enumerate(dataloader):

        tensors = tensors.float().to(device)
        labels = labels.to(device)

        outputs = model(tensors)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        #if max_abs_grad:
        #    torch.nn.utils.clip_grad_value_(model.parameters(), max_abs_grad)


        optimizer.step()

        if scheduler:
            scheduler.step()

        #exponential moving evaraging of loss
        avg_loss = beta * avg_loss + (1-beta)*loss.item()
        smoothed_loss = avg_loss / (1 - beta**(itr_idx+1))

        #outputs = softmax(outputs)
        outputs = outputs.cpu().tolist()
        labels = labels.cpu().tolist()

        current_predictions = list(zip(outputs, labels))
        all_predictions.extend(current_predictions)

        if not silent:
            pbar.update(1)
            pbar.set_description(f"Running loss:{smoothed_loss:.4}")

    if not silent:
        del pbar

    return smoothed_loss, all_predictions

def model_eval(model, optimizer, dataloader, device, use_tuplemax=True, silent=False):

    model.eval() #model to evaluation mode

    if use_tuplemax:
        criterion = losses.TuplemaxLoss()
    else:
        criterion = nn.CrossEntropyLoss() #binary cross-entropy

    if not silent:
        tot_itr = len(dataloader)//dataloader.batch_size #total evaluation iterations
        pbar = tqdm(total = tot_itr, ncols=700) #progress bar

    all_loss = 0. #all losses, for simple averaging

    all_predictions = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():

        for itr_idx, (tensors, labels) in enumerate(dataloader):

            tensors = tensors.float().to(device)
            labels = labels.to(device)

            outputs = model(tensors)

            loss = criterion(outputs, labels)

            all_loss += loss.item()

            #outputs = softmax(outputs)
            outputs = outputs.cpu().tolist()
            labels = labels.cpu().tolist()

            current_predictions = list(zip(outputs, labels))
            all_predictions.extend(current_predictions)

            if not silent:
                pbar.update(1)
                pbar.set_description(f"Running loss:{all_loss/(itr_idx+1):.4}")

    if not silent:
        del pbar

    return all_loss/(itr_idx+1), all_predictions