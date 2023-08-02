# import libraries and functions
from functions_new import *
import torch
from itertools import islice
from matplotlib import pyplot as plt
import numpy as np
from csv import DictWriter
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import csv
import math 
import os
from os import listdir
import pickle
import pandas as pd
from tqdm.notebook import tqdm
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from numpy import savez_compressed, asarray
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, T5EncoderModel, T5Tokenizer
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence
import requests
from torch.autograd import Variable
from sklearn import metrics
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler
import re
import gc
import warnings
from evcouplings.align import Alignment, map_matrix,read_fasta
from collections import OrderedDict

#import Pre-Trained model
tokenizer = T5Tokenizer.from_pretrained("/scratch/brussel/104/vsc10400/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("/scratch/brussel/104/vsc10400/prot_t5_xl_uniref50")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

# Import mutations
#all_mutations =pd.read_csv('DRGN_mutations_012023.csv')
#all_mutations =pd.read_csv('P10398_all.csv')
#all_mutations = pd.read_csv('balanced_somatic_germline_missense_noconflicts.csv')
#all_mutations = all_mutations[all_mutations['origin'] == 'germline']
all_mutations = pd.read_csv('Tier1_2_3_common_balanced+-2.csv')
all_mutations = all_mutations.sort_values(by='uniprot')
#all_mutations = all_mutations.sort_values(by='uniprot id') # use for DRGN set and single gene calculation
all_mutations = all_mutations.reset_index(drop=True)
print(all_mutations.head())

m = nn.MaxPool1d(50) # Max Pooling for reduction of features from 1024 to 20 per AA
count = 0
ind_excl_tier = []
for i, mut in all_mutations.iterrows():
    if len(mut['mut_sequence']) <= 2200:
        count += 1
    else:
        ind_excl_tier.append(i)

all_mutations = all_mutations.drop(all_mutations.index[ind_excl_tier])

print(len(all_mutations))
#list_uniprot = all_mutations['uniprot id'].unique() #DRGN set
list_uniprot = all_mutations['uniprot'].unique() # Tier1_2_3_humsavar
print('Unique transcripts on set:', len(list_uniprot))

curwd = os.getcwd()
msa_path= str(curwd) + '/all_msas/'

for uniprot in list_uniprot:
  print(uniprot)

  #mut_gene = all_mutations[all_mutations['uniprot id'] == uniprot] #DRGN set
  mut_gene = all_mutations[all_mutations['uniprot'] == uniprot] # Tier1_2_3_CGI_humsavar
  ## Read in a sequence alignment from a fasta file
  if os.path.isfile(msa_path + uniprot+ ".a3m"): # True if file exists
    name_msa_file = msa_path + uniprot+ ".a3m"
  else:
    print('MSA not found in folder !')
    continue

  ### MSA of gene
  aln_subsection = msa_protocol(name_msa_file)
  if aln_subsection.L > 2200:
    #print('bigger than 500 AA')
    continue

  ### Protrans
  # calculate the ProTrans for WT protein
  lines_list = []
  for line in range(len(aln_subsection)):
    temp = aln_subsection.matrix[line, :].tolist()

    x = [x.upper() for x in temp]
    lines_list.append(x)

  str1 = " "
  lines_string = [str1.join(first_line) for first_line in lines_list]

  sequences_WT = [re.sub(r"[-.]", "X", sequence) for sequence in lines_string]

  indices_to_excl = []
  seq_pooled = []
  if aln_subsection.L <501:
      BATCH_FILE_SIZE = 15
  else:
      BATCH_FILE_SIZE = 1

  test_features_WT = []
  for count in range(0, math.floor(len(sequences_WT) / BATCH_FILE_SIZE)):
      i = sequences_WT[count*BATCH_FILE_SIZE:(count+1)*BATCH_FILE_SIZE][:]
      ids = tokenizer.batch_encode_plus(i, add_special_tokens=True, padding='longest')
      input_ids = torch.tensor(ids['input_ids']).to(device)
      attention_mask = torch.tensor(ids['attention_mask']).to(device)

      with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        #print(embedding.shape)
        for seq_num in range(len(embedding)):
          seq_len = (attention_mask[seq_num] == 1).sum()
          seq_emd = embedding[seq_num][:seq_len-1]
          test_features_WT.append(seq_emd)
      del attention_mask
      gc.collect()

  # converting list to array
  arr_WT = np.array(test_features_WT)

  seq_temp = torch.tensor(arr_WT)
  arr_WT = m(seq_temp) # use when you want to reduce dimensions from 1024 to 20
  arr_WT =arr_WT.numpy()

  columns = range(0, arr_WT.shape[1])
  differences_WT = []
  densities_WT = []
  #density_threshold_WT = []
  for col in columns:
      first_col = arr_WT[:, col]
      gmm = GaussianMixture(n_components=1).fit(first_col)
      densities_temp = gmm.score_samples(first_col)
      densities_WT.append(densities_temp)
      threshold_temp = np.percentile(densities_temp, 1)
      #density_threshold_WT.append(threshold_temp)
      #differences_WT.append(abs(densities_temp[0] - threshold_temp) )
      differences_WT.append(densities_temp[0] - threshold_temp)

  ### Calculate differences of all mutations of gene
  #print(len(mut_gene))

  for k, mut in mut_gene.iterrows():
    diction_test = {} # dictionary containing the difference of log-probabilities of mutation from the lof-prob of WT

    mut_seq = mut['mut_sequence']# mutated sequence
    position = int(mut['position'])-1
    AA_orig = mut['AA_orig']
    AA_targ = mut['AA_targ']

    new_str = [str(x) for x in mut_seq]
    new_str[position] = AA_targ

    str1 = " "
    lines_string = str1.join(new_str)
    MUT_sequence = re.sub(r"[-.]", "X", lines_string)

    ids = tokenizer.batch_encode_plus([MUT_sequence], add_special_tokens=True, padding='longest')
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
      embedding = model(input_ids=input_ids,attention_mask=attention_mask)
      embedding = embedding.last_hidden_state.cpu().numpy()
      #print(embedding.shape)
      seq_len = (attention_mask == 1).sum()
      seq_emd = embedding[:, :seq_len-1, :]


    seq_emd = torch.tensor(seq_emd)
    seq_emd = m(seq_emd) # use when you want to reduce dimensions from 1024 to 20
    seq_emd =seq_emd.numpy()

    arr_WT[0] = seq_emd[0]
    del embedding, ids, MUT_sequence, attention_mask
    gc.collect()

    columns = range(0, arr_WT.shape[1])
    differences_MUT = []
    densities_MUT = []
    #density_threshold_MUT = []
    for col in columns:
        first_col = arr_WT[:, col]
        gmm = GaussianMixture(n_components=1).fit(first_col)
        densities_temp = gmm.score_samples(first_col)
        densities_MUT.append(densities_temp)
        threshold_temp = np.percentile(densities_temp, 1)
        #density_threshold_MUT.append(threshold_temp)
        #differences_MUT.append(abs(densities_temp[0] - threshold_temp) )
        differences_MUT.append(densities_temp[0] - threshold_temp)

    dif_dif = [differences_WT[i] - differences_MUT[i] for i in range(len(differences_MUT))] # difference of WT and Mutated sequence log probabiities

    ### Save dif_dif in a dictionary with keys: genename_mutation
    gene_name_mutation = uniprot +'_' + AA_orig + str(mut['position']) + AA_targ
    diction_test['uniprot_mut'] = gene_name_mutation
    diction_test['Log dif'] = dif_dif
    diction_test['label'] = mut['label']

    # list of column
    field_names = ['uniprot_mut','Log dif', 'label']
    # Open your CSV file in append mode
    # Create a file object for this file
    with open('log_probWT_MUT_Tier1_2_3_common_balanced+-22200AA_57maxpool.csv', 'a') as f_object:
	# Pass the file object and a list
        # of column names to DictWriter()
        # You will get a object of DictWriter
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        #Pass the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(diction_test)
        #Close the file object
        f_object.close()

    del dif_dif, diction_test
    torch.cuda.empty_cache()
    gc.collect()
