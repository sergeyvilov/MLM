#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import os
import sys

from sklearnex import patch_sklearn
patch_sklearn()

import sklearn

import sklearn.pipeline 
import sklearn.model_selection
import sklearn.metrics

from sklearn.preprocessing import StandardScaler

from multiprocessing import Pool

import optuna

sys.path.append("/data/ouga/home/ag_gagneur/l_vilov/workspace/species-aware-DNA-LM/mpra_griesemer/utils") 

from models import *
from misc import dotdict

import scipy.stats


# In[2]:


data_dir = '/s/project/mll/sergey/effect_prediction/MLM/siegel_2022/'


# In[3]:

import argparse

parser = argparse.ArgumentParser("main.py")

input_params = dotdict({})

parser.add_argument("--cell_type", help = "Beas2B or Jurkat", type = str, required = True)

parser.add_argument("--response", help = "steady_state or stability", type = str, required = True)

parser.add_argument("--model", help = 'embedding name, can be "MLM" "word2vec" "effective_length" or "Nmers" where N is an integer', type = str, required = True)

parser.add_argument("--output_dir", help = 'output folder', type = str, required = True)

parser.add_argument("--N_trials", help = "number of optuna trials", type = int, default = 100, required = False)

parser.add_argument("--keep_first", help = "perform hpp search only at the first split, then use these hyperparameters", action='store_true', default = False, required = False)

parser.add_argument("--N_splits", help = "number of GroupShuffleSplits", type = int, default = 200, required = False)

parser.add_argument("--N_CVsplits", help = "number of CV splits for hyperparameter search", type = int, default = 5, required = False)

parser.add_argument("--seed", help = "seed fot GroupShuffleSplit", type = int, default = 1, required = False)

input_params = vars(parser.parse_args())

input_params = dotdict(input_params)

for key,value in input_params.items():
    print(f'{key}: {value}')
    
# In[4]:


mpra_df = pd.read_csv(data_dir + f'{input_params.cell_type}.tsv', sep='\t') #sequence info

mlm_embeddings = np.load(data_dir + "embeddings/seq_len_5000/embeddings.npy") #masked language model embeddings

#Data Cleaning
# Take only SNP mutations
# Remove nan values in Expression column

if input_params.response == 'steady_state':
    mpra_df['Expression'] = mpra_df.ratios_T0_GC_resid
elif input_params.response == 'stability':
    mpra_df['Expression'] = mpra_df.ratios_T4T0_GC_resid



regions_utr_map = pd.read_csv(data_dir + 'regions_hg38/regions_3UTR_GRCh38.bed', sep='\t',
                             names = ['region_start','region_end','ids','utr_start','utr_end','strand']) #mapping between regions and 3'UTR coordinates

regions_utr_map = regions_utr_map[(regions_utr_map.region_start>=regions_utr_map.utr_start) & 
    (regions_utr_map.region_end<=regions_utr_map.utr_end)].drop_duplicates() #region should be entirely within 3'UTR

regions_utr_map = regions_utr_map.drop_duplicates(subset='ids', keep=False) #remove variants assign to multiple 3'UTR

regions_utr_map['stop_codon_dist'] = regions_utr_map.apply(lambda x: x.region_end-x.utr_start 
                      if x.strand=='+' else x.utr_end - x.region_start, axis=1)  #distance to the stop codon, must be below 5000 for MLM

mpra_df = mpra_df.merge(regions_utr_map[['ids','stop_codon_dist']], how='left')

mpra_df.drop_duplicates(inplace=True)



flt = (mpra_df.Expression.isna()) | (mpra_df.ARE_length_perfect.isna()) | (mpra_df.stop_codon_dist.isna()) | (mpra_df.stop_codon_dist>5000) | (~mpra_df.issnp.astype(bool))

mpra_df = mpra_df[~flt]

mlm_embeddings = mlm_embeddings[mpra_df.index]


# In[5]:


mpra_df['group'] = mpra_df.region.apply(lambda x:x.split('|')[1].split(':')[0])


if input_params.model=='MLM':

    X = mlm_embeddings

elif 'mers' in input_params.model:
    
    k = int(input_params.model[0])
        
    kmerizer = Kmerizer(k=k)
    X = np.stack(mpra_df.seq.apply(lambda x: kmerizer.kmerize(x))) 
        
elif input_params.model=='word2vec':
        
    X = word2vec_model(mpra_df)

elif input_params.model=='effective_length':
    
    X = mpra_df.ARE_registration_perfect + mpra_df.ARE_length_perfect
    X = np.expand_dims(X.values,1)

#X = np.hstack((X,np.expand_dims(mpra_df.min_free_energy.values,axis=1)))

y = mpra_df['Expression'].values
groups = mpra_df['group'].values


# In[8]:


def hpp_search(X,y,groups,cv_splits = 5):
    
    '''
    Perform Hyperparameter Search using OPTUNA Bayesian Optimisation strategy
    
    The bets hyperparameters should maximize coefficient of determination (R2)
    
    The hyperparameter range should first be adjused with grid search to make the BO algorithm converge in reasonable time
    '''

    def objective(trial):

        C = trial.suggest_float("C", 1e-2, 1, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-5, 1, log=True)
        gamma = trial.suggest_float("gamma", 1e-5, 1, log=True)

        clf = sklearn.svm.SVR(C=C, epsilon=epsilon, gamma=gamma)

        pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),clf)
        
        cv_score = sklearn.model_selection.cross_val_score(pipe, X, y, groups=groups, 
                     cv = sklearn.model_selection.GroupKFold(n_splits = cv_splits), scoring = 'r2', n_jobs = -1)
        
        av_score = cv_score.mean()
        
        return av_score
    
    study = optuna.create_study(direction = "maximize")

    study.optimize(objective, n_trials = input_params.N_trials)
    
    best_params = study.best_params
    
    return best_params


# In[ ]:

gss = sklearn.model_selection.LeaveOneGroupOut() 

train_idx, _ = next(iter(gss.split(X, y, groups)))

best_hpp = hpp_search(X[train_idx],y[train_idx],groups[train_idx],cv_splits = input_params.N_CVsplits) #get optimal hyperparameters

#best_hpp = {'C': 0.03943153578419499, 'epsilon': 0.0712140417882623, 'gamma': 0.000232694021502066}

def apply_regression(args):
    
    train_idx, test_idx = args

    #predict with SVR
    
    pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),
                                              sklearn.svm.SVR(**best_hpp))
        
    pipe.fit(X[train_idx],y[train_idx])  
    
    y_pred_svr = pipe.predict(X[test_idx])  
    
    #predict with Lasso
    #use inner CV loop to adjust alpha
    
    group_kfold = sklearn.model_selection.GroupKFold(n_splits=input_params.N_CVsplits).split(X[train_idx],y[train_idx],groups[train_idx])
    
    pipe_lasso = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), sklearn.linear_model.LassoCV(cv=group_kfold, alphas=10.**np.arange(-6,0))) 
    
    pipe_lasso.fit(X[train_idx],y[train_idx])
    
    y_pred_lasso = pipe_lasso.predict(X[test_idx])
        
    print('done')

    return list(zip(mpra_df.ids.iloc[test_idx], y_pred_svr, y_pred_lasso))
 
def run_pool():
    
    all_res = []
    
    pool = Pool(processes=input_params.n_jobs,maxtasksperchild=5)

    for res in pool.imap(apply_regression,gss.split(X,y,groups)):
        all_res.extend(res)
     
    pool.close()
    pool.join()
    
    return all_res

print('running parallel')

all_res = run_pool()

all_res = pd.DataFrame(all_res, columns=['ids','y_pred_svr','y_pred_lasso'])

mpra_df.merge(all_res, how='left').to_csv(input_params.output_dir + f'/{input_params.cell_type}-{input_params.response}-{input_params.model}.tsv', sep='\t', index=None) 