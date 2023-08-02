# Libraries
import csv
from sklearn.metrics import precision_recall_fscore_support
from evcouplings.align import Alignment, map_matrix, read_fasta
import seaborn as sns
from scipy.stats import wilcoxon
from collections import OrderedDict
import pickle
from csv import DictWriter
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
from collections import Counter
import gc
import warnings
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import matthews_corrcoef
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
from sklearn.preprocessing import MinMaxScaler , StandardScaler   
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from numpy import asarray,savez_compressed
import requests
from sklearn import metrics
import re
import os
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, T5EncoderModel, T5Tokenizer
import torch.nn as nn
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from sklearn.neighbors import KernelDensity

"""# Functions"""
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def notNaN(num):
    return num == num


def Average(lst):
    return sum(lst) / len(lst)


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def read_a3m(fileobj, inserts="first"):
    """
    Read an alignment in compressed a3m format and expand
    into a2m format.
    .. note::
        this function is currently not able to keep inserts in all the sequences
    ..todo::
        implement this
    Parameters
    ----------
    fileobj : file-like object
        A3M alignment file
    inserts : {"first", "delete"}
        Keep inserts in first sequence, or delete
        any insert column and keep only match state
        columns.
    Returns
    -------
    OrderedDict
        Sequences in alignment (key: ID, value: sequence),
        in order they appeared in input file
    Raises
    ------
    ValueError
        Upon invalid choice of insert strategy
    """
    seqs = OrderedDict()

    for i, (seq_id, seq) in enumerate(read_fasta(fileobj)):
        # remove any insert gaps that may still be in alignment
        # (just to be sure)
        seq = seq.replace(".", "")

        if inserts == "first":
            # define "spacing" of uppercase columns in
            # final alignment based on target sequence;
            # remaining columns will be filled with insert
            # gaps in the other sequences
            if i == 0:
                uppercase_cols = [
                    j for (j, c) in enumerate(seq)
                    if (c == c.upper() or c == "-")
                ]
                gap_template = np.array(["."] * len(seq))
                filled_seq = seq
            else:
                uppercase_chars = [
                    c for c in seq if c == c.upper() or c == "-"
                ]
                filled = np.copy(gap_template)
                filled[uppercase_cols] = uppercase_chars
                filled_seq = "".join(filled)

        elif inserts == "delete":
            # remove all lowercase letters and insert gaps .;
            # since each sequence must have same number of
            # uppercase letters or match gaps -, this gives
            # the final sequence in alignment
            seq = "".join([c for c in seq if c == c.upper() and c != "."])
        else:
            raise ValueError(
                "Invalid option for inserts: {}".format(inserts)
            )

        seqs[seq_id] = seq

    return seqs


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def msa_protocol(name_msa_file):
  ### name_msa_file: path to msa fime of gene
  ### return: alignment of gene

  with open(name_msa_file, "r") as infile:
    #seqs = OrderedDict()
    next(infile)

    #for i, (seq_id, seq) in enumerate(read_fasta(infile)):
    proper_infile = read_a3m(infile, inserts = "delete") # convert from a3m to a2m
    #seq = seq.replace(".", "")
    # seqs[seq_id] = seq
    #n_items = take(n, seqs.items())

    #aln = Alignment.from_file(proper_infile, format="fasta")
    aln = Alignment.from_dict(proper_infile)

  # Sequence length and number of sequences
  print(f"alignment is of length {aln.L} and has {aln.N} sequences")

  # Protocol Hopf
  # calculate the percent identity of every sequence in the alignment to the first sequence
  ident_perc = aln.identities_to(aln.matrix[0])
  ident_perc_list = ident_perc.tolist()

  # keep identifiers with > 50 percentage identity and colunns with at least 70% occupancy
  index_keep = []
  for i, iden in enumerate(ident_perc_list):
    if iden > 0.5: # 0.5= sequences with at least 50% of identity to the frst sequence are kept
      index_keep.append(i)

  #use the "count" method of the class  -  Percentage of gaps
  maximum1 = aln.count(axis="seq",char="-")#.argmax()

  filtered_ind = [i for i in range(len(maximum1)) if maximum1[i] <= 0.3] # 0.3 30% of gaps
  sequences_to_keep = intersection(index_keep, filtered_ind) # keep indeces that satisfy both conditions

  selection_index = sequences_to_keep
  aln_subsection = aln.select(sequences=selection_index)
  print(f"the new alignment has {aln_subsection.N} sequences")

  # if remaining sequences in MSA < 15 redo the process with less strict filtering
  if aln_subsection.N <15:
    index_keep = []
    for i, iden in enumerate(ident_perc_list):
      #if iden > 0.05: # 0.3= sequences with at least 5% of identity to the frst sequence are kept
      if iden > 0.27: # 0.3= sequences with at least 10% of identity to the frst sequence are kept
        index_keep.append(i)
    filtered_ind = [i for i in range(len(maximum1)) if maximum1[i] <= 0.7] # max 60% of gaps
    sequences_to_keep = intersection(index_keep, filtered_ind) # keep indeces that satisfy both conditions
    selection_index = sequences_to_keep
    aln_subsection = aln.select(sequences=selection_index)
 
  print(f"the new alignment has {aln_subsection.N} sequences")
  return aln_subsection

def unique(list1):
 
    # initialize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    print(f'{len(unique_list)} unique transcripts')
    return unique_list


def cancer_labels(val):
    if 'cancer' in val:
        return 'cancer'
    else:
        return 'non_canceric'

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    return heatmap.T, extent



"""Confidence score"""

def calculation_WT_MUT(uniprot, all_mutations, msa_path): 
  threshold_WT_benign_predictions_in, threshold_WT_pathogenic_predictions_in, threshold_MUT_pathogenic_predictions_in, threshold_MUT_benign_predictions_in= [], [], [], [] # for all proteins
  print(uniprot)
  print(len(all_mutations))
  aic_in, mutations_in, log_prob_in =[], [], []

  mut_gene = all_mutations 

  ## Read in a sequence alignment from a fasta file
  name_msa_file = msa_path + uniprot+ ".a3m"

  ### MSA of gene
  aln_subsection = msa_protocol(name_msa_file)

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

  ### Calculate differences of all mutations of gene
  for k, mut in mut_gene.iterrows():
    
    diction_test = {} # dictionary containing the difference of log-probabilities of mutation from the lof-prob of WT

    mut_seq = mut['mut_sequence']# mutated sequence
    position = int(mut['position'])-1
    AA_orig = mut['AA_orig']
    AA_targ = mut['AA_targ']

    temp_mut = AA_orig + str(mut['position']) + AA_targ 

    columns = range(0, arr_WT.shape[1])
    for col in columns:
        if col == position:
            first_col = arr_WT[:, col]
            gmm = GaussianMixture(n_components=1).fit(first_col)
            densities_temp = gmm.score_samples(first_col)
            threshold_temp = np.percentile(densities_temp, 1)
            averages_log_prob = round(Average(densities_temp), 3) # include the WT in the log-probability calculation
            aic_metric= round(gmm.aic(first_col),3) # the lower the better
            break

    if mut['D2Deep_prediction']>=0.5 : # if pathogenic but incorrect
          threshold_WT_pathogenic_predictions_in.append(densities_temp[0] - threshold_temp) 
    elif mut['D2Deep_prediction']<0.5 : # if benign but correct
          threshold_WT_benign_predictions_in.append(densities_temp[0] - threshold_temp) 

    #deep copy of WT array
    arr_WT_MUT = arr_WT.copy()

    # Mutant - threshold
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
      seq_len = (attention_mask == 1).sum()
      seq_emd = embedding[:, :seq_len-1, :]

    seq_emd = torch.tensor(seq_emd)
    seq_emd = m(seq_emd) # use when you want to reduce dimensions from 1024 to 20
    seq_emd =seq_emd.numpy()

    arr_WT_MUT[0] = seq_emd[0]
    del embedding, ids, MUT_sequence, attention_mask
    gc.collect()

    columns = range(0, arr_WT_MUT.shape[1])

    for col in columns:
        if col == position:
            first_col = arr_WT_MUT[:, col]
            gmm = GaussianMixture(n_components=1).fit(first_col)
            densities_temp = gmm.score_samples(first_col)
            threshold_temp = np.percentile(densities_temp, 1)
            break
     	        
    if mut['D2Deep_prediction']>=0.5 : # if pathogenic but incorrect
        threshold_MUT_pathogenic_predictions_in.append(densities_temp[0] - threshold_temp) 
    elif mut['D2Deep_prediction']<0.5 : # if benign but correct
        threshold_MUT_benign_predictions_in.append(densities_temp[0] - threshold_temp)

 
    mutations_in.append(temp_mut) # append mutation
    log_prob_in.append(averages_log_prob) # append log-probability of MSA position
    aic_in.append(aic_metric)

  return aic_in, log_prob_in, mutations_in, threshold_WT_benign_predictions_in, threshold_MUT_benign_predictions_in, threshold_WT_pathogenic_predictions_in, threshold_MUT_pathogenic_predictions_in 

#import ProtT5Pre-Trained model
tokenizer = T5Tokenizer.from_pretrained("/scratch/brussel/104/vsc10400/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("/scratch/brussel/104/vsc10400/prot_t5_xl_uniref50")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()
gc.collect()

curwd = os.getcwd()
pardir = os.path.abspath(os.path.join(curwd, os.pardir))
msa_path= str(pardir) + '/all_msas/'
m = nn.MaxPool1d(50) # Max Pooling for reduction of features from 1024 to 50 per AA

#threshold_WT_benign_predictions, threshold_WT_pathogenic_predictions, threshold_MUT_pathogenic_predictions, threshold_MUT_benign_predictions= [], [], [], [] # for all proteins

#genes= ['PTEN', 'AR', 'BRAF','TP53', 'CHEK2' ]
genes= ['P00519', 'Q6UWZ7', 'Q04771', 'Q9NZK5', 'O43918', 'P31749', 'P31751', 'Q9Y243', 'Q9UPS8', 'Q9H6X2', 'P10398', 'Q99728', 'O14757', 'P07333', 'P35222', 'P41212', 'P23468']
for gene in genes:
  
  threshold_WT_benign_predictions, threshold_WT_pathogenic_predictions, threshold_MUT_pathogenic_predictions, threshold_MUT_benign_predictions= [], [], [], [] # for all proteins
  filename= gene+ "_d2d_results.csv"

  pten_predictions = pd.read_csv(filename, sep=',')
  # for converting d2d_results to d2d_performance_vs_clinvar format
  temp = pten_predictions.mutation.str.split(pat='_',expand=True)[0]
  pten_predictions['uniprot id'] = temp
  temp = pten_predictions.mutation.str.split(pat='_',expand=True)[1]
  pten_predictions= pten_predictions.drop('mutation', axis=1)
  pten_predictions['mutation']= temp
  pten_predictions['AA_orig'] = pten_predictions['mutation'].str[:1]
  pten_predictions['AA_targ'] = pten_predictions['mutation'].str[-1:]
  pten_predictions['position'] = pten_predictions['mutation'].str[1:-1]
  # drop original column
  pten_predictions = pten_predictions.drop('mutation', axis=1)
  pten_predictions = pten_predictions.rename(columns={"prediction": "D2D_prediction"})

  all_mutations = pten_predictions
  list_uniprot = all_mutations.iloc[0]['uniprot id']

  # confidence B
  aic_temp, log_prob_temp, mutations, threshold_WT_benign_predictions_temp, threshold_MUT_benign_predictions_temp, threshold_WT_pathogenic_predictions_temp, threshold_MUT_pathogenic_predictions_temp = calculation_WT_MUT(list_uniprot, all_mutations, msa_path)

  # create mutation, probability to be benign, probability to be pathogenic, WT_thres, MUT_thres
  confidence_df = pd.DataFrame(list(zip(mutations, aic_temp, log_prob_temp)), columns = ['mutation', 'AIC', 'Log_prob'])
  confidence_df.to_csv(gene+'_confidenceB.csv') 

# Visualizations
'''
# plot results  
data = [threshold_WT_benign_predictions, threshold_WT_pathogenic_predictions]
fig = plt.figure(figsize =(10, 10))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data)
ax.grid(True, 'Major', 'y')
plt.savefig('TP53_BRAF_AR_CHEK2_PTEN_all_WT_benign_pathogenic.png')
plt.clf()

data = [threshold_MUT_benign_predictions, threshold_MUT_pathogenic_predictions]
fig = plt.figure(figsize =(10, 10))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data)
ax.grid(True, 'Major', 'y')
 
plt.savefig('TP53_BRAF_AR_CHEK2_PTEN_all_MUT_benign_pathogenic.png')
plt.clf()

plt.rcParams["figure.figsize"] = (10,10)
plt.scatter(threshold_WT_benign_predictions, threshold_MUT_benign_predictions, alpha=0.6, c='#388ed1') # blue
#plt.scatter(threshold_WT_pathogenic_correct, threshold_MUT_pathogenic_correct, alpha=0.6, c='#d74e26') # red
#plt.scatter(threshold_WT_benign_incorrect, threshold_MUT_benign_incorrect,alpha=0.6, c= '#388ed1')# c='#006837') # green
plt.scatter(threshold_WT_pathogenic_predictions, threshold_MUT_pathogenic_predictions, alpha=0.6, c='#d74e26')#,c='#f7931e') # orange
plt.xlabel('WT - threshold')
plt.ylabel('MUT - threshold')
plt.title('TP53_BRAF_AR_CHEK2_PTEN_all_WT_MUT_predictions')

#plt.savefig('WT_MUT_easy.png')
plt.savefig('TP53_BRAF_AR_CHEK2_PTEN_all_WT_MUT_predictions.png')
plt.clf()


#benign
# Add the scatter plots on top with transparency alpha=0.6
plt.scatter(threshold_WT_benign_predictions, threshold_MUT_benign_predictions, alpha=0.6, c='#388ed1')
plt.xlabel('Threshold WT predictions')
plt.ylabel('Threshold MUT predictions')
plt.title('benign predictions')
plt.savefig('TP53_BRAF_AR_CHEK2_PTEN_all_WT_MUT_benign predictions.png')
plt.clf()
#heatmap
img, extent = myplot(threshold_WT_benign_predictions,threshold_MUT_benign_predictions, 2)
plt.imsave('TP53_BRAF_AR_CHEK2_PTEN_all_WT_MUT_benign_heatmap.png',img)
plt.clf()

#pathogenic
plt.scatter(threshold_WT_pathogenic_predictions, threshold_MUT_pathogenic_predictions, alpha=0.6, c='#d74e26')
plt.xlabel('Threshold WT predictions')
plt.ylabel('Threshold MUT predictions')
plt.title('Pathogenic threshold predictions')
plt.savefig('TP53_BRAF_AR_CHEK2_PTEN_all_WT_MUT_pathogenic_predictions.png')
plt.clf()
#heatmap
img, extent = myplot(threshold_WT_pathogenic_predictions,threshold_MUT_pathogenic_predictions, 2)
plt.imsave('TP53_BRAF_AR_CHEK2_PTEN_all_WT_MUT_pathogenic_heatmap.png',img)
plt.clf()


# features
features_pathogenic = [element1 - element2 for (element1, element2) in zip(threshold_WT_pathogenic_predictions ,threshold_MUT_pathogenic_predictions)]
features_benign = [element1 - element2 for (element1, element2) in zip(threshold_WT_benign_predictions ,threshold_MUT_benign_predictions)]

data = [features_benign, features_pathogenic]
fig = plt.figure(figsize =(10, 7))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(data)
ax.grid(True, 'Major', 'y')
plt.savefig('TP53_BRAF_AR_CHEK2_PTEN_all_features_benign_pathogenic.png')
plt.clf()

# Scatter plot + histograms
x1 = threshold_WT_benign_predictions
y1 = threshold_MUT_benign_predictions

x2 = threshold_WT_pathogenic_predictions
y2 = threshold_MUT_pathogenic_predictions


# alternate points
# Create scatter plot with the alternate x and y values and colors

X1 = np.column_stack((x1,y1))
X2 = np.column_stack((x2,y2))
data = np.concatenate([X1,X2])

classes = np.concatenate([np.repeat('X', X1.shape[0]),
                          np.repeat('X2', X2.shape[0])])

plot_idx = np.random.permutation(data.shape[0])
colors,labels = pd.factorize(classes)
colors = np.array(['blue', 'red'])[pd.factorize(classes)[0]]

fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3])
ax1 = plt.subplot(gs[1, 0])
ax2 = plt.subplot(gs[0, 0])
ax3 = plt.subplot(gs[1, 1])

#ax1.scatter(x1, y1, s=10, alpha=0.5, c='blue', label='Benign') # when random (alternate) point dispay not used
#ax1.scatter(x2, y2, s=10, alpha=0.5, c='red', label='Pathogenic')
ax1.scatter(data[plot_idx, 0], data[plot_idx, 1], c=colors[plot_idx], alpha=0.4) # when random (alternate) point display is used
ax1.set_xlabel('WT-threshold')
ax1.set_ylabel('MUT-threshold')
ax1.set_title('Scatter plot')
ax1.legend()

# plot the histogram for the X variable of both clusters
ax2.hist(x1, bins=30, alpha=0.5, color='blue', label='Benign', orientation='vertical')
ax2.hist(x2, bins=30, alpha=0.5, color='red', label='Pathogenic', orientation='vertical')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('WT-threshold')
ax2.set_title('Histogram for WT-threshold')
ax2.legend()


# plot the histogram for the Y variable of both clusters
ax3.hist(y1, bins=30, alpha=0.5, color='blue', label='Benign', orientation='horizontal')
ax3.hist(y2, bins=30, alpha=0.5, color='red', label='Pathogenic', orientation='horizontal')
ax3.set_xlabel('MUT-threshold')
ax3.set_ylabel('Frequency')
ax3.set_title('Histogram for MUT-threshold')
ax3.legend()

# adjust the layout
plt.tight_layout()

plt.savefig('new_TP53_BRAF_AR_CHEK2_PTEN_scatter_histograms.png')
plt.clf()
'''
