import pickle
import time
import builtins
import sys
import torch
import os

class dotdict(dict):
    '''
    dot.notation access to dictionary attributes
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def list2range(v):
    r = []
    for num in v:
      if not ':' in num:
        r.append(int(num))
      else:
        k = [int(x) for x in num.split(':')]
        if len(k)==2:
            r.extend(list(range(k[0],k[1]+1)))
        else:
            r.extend(list(range(k[0],k[1]+1,k[2])))
    return r


def save_predictions(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def print(*args, **kwargs):
    '''
    Redefine print function for logging
    '''
    now = time.strftime("[%Y/%m/%d-%H:%M:%S]-", time.localtime()) #current date and time at the beggining of each printed line
    builtins.print(now, *args, **kwargs)
    sys.stdout.flush()

def save_model_weights(model, optimizer, output_dir, epoch):
    '''
    Save model and optimizer weights
    '''
    config_save_base = os.path.join(output_dir, f'epoch_{epoch}_weights')

    print(f'EPOCH:{epoch}: SAVING MODEL, CONFIG_BASE: {config_save_base}\n')

    torch.save(model.state_dict(), config_save_base+'_model') #save model weights

    torch.save(optimizer.state_dict(), config_save_base+'_optimizer') #save optimizer weights