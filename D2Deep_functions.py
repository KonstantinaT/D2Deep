import pandas as pd
import re
import os
import shutil
import copy
import csv
import numpy as np
import seaborn as sns
import pickle
import gc
import requests
from sklearn.metrics import precision_recall_fscore_support
from evcouplings.align import Alignment, map_matrix, read_fasta
import seaborn as sns
from scipy.stats import wilcoxon
from collections import OrderedDict, Counter
from csv import DictWriter
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import warnings
from tqdm.notebook import tqdm
from sklearn.metrics import matthews_corrcoef
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from numpy import asarray,savez_compressed
from sklearn import metrics
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, T5EncoderModel, T5Tokenizer
import torch.nn as nn
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from sklearn.neighbors import KernelDensity
import random


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
  #print(f"alignment is of length {aln.L} and has {aln.N} sequences")


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
  #print(f"the new alignment has {aln_subsection.N} sequences")

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

  if aln_subsection.N <15:
    index_keep = []
    for i, iden in enumerate(ident_perc_list):
      #if iden > 0.05: # 0.3= sequences with at least 5% of identity to the frst sequence are kept
      if iden > 0.2: # 0.3= sequences with at least 20% of identity to the frst sequence are kept
        index_keep.append(i)
    filtered_ind = [i for i in range(len(maximum1)) if maximum1[i] <= 0.7] # max 60% of gaps
    sequences_to_keep = intersection(index_keep, filtered_ind) # keep indeces that satisfy both conditions
    selection_index = sequences_to_keep
    aln_subsection = aln.select(sequences=selection_index)

  #print(f"the new alignment has {aln_subsection.N} sequences")
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


def normalise_confidence(gene_confidence):
  # min max
  max_log = max(gene_confidence['Log_prob'].tolist())
  min_log = min(gene_confidence['Log_prob'].tolist())
  gene_confidence['log_normalized'] = (gene_confidence['Log_prob'] - min_log )/(max_log - min_log)
  condition1 = gene_confidence['D2Deep_prediction'] >= 0.5 # for the 5 initial genes TP53, PTEN, AR, BRAF and ChEK2: D2D_prediction
  condition2 = (gene_confidence['log_normalized'] >= 0.5) & (gene_confidence['D2Deep_prediction'] < 0.5)
  condition3 = (gene_confidence['log_normalized'] < 0.5) & (gene_confidence['D2Deep_prediction'] < 0.5)

  # when using only log-GMM
  gene_confidence.loc[condition1, 'overall_confidence'] = gene_confidence.loc[condition1, 'log_normalized']   # Set values in 'B' as half of values in 'C' when the condition is met
  gene_confidence.loc[condition2, 'overall_confidence'] = abs(1- gene_confidence.loc[condition2, 'log_normalized'] ) # *1.2#Set values in 'B' as half of values in 'D' when the condition is not met
  gene_confidence.loc[condition3, 'overall_confidence'] = 1- gene_confidence.loc[condition3, 'log_normalized']*1.3

  return gene_confidence

class Classifier2L(nn.Module):
    def __init__(self, hidden, hidden2, dropout=0):
        super(Classifier2L, self).__init__()
        self.hidden = hidden
        self.hidden2 = hidden2
        self.num_feature = 2200
        self.dropout = dropout
        self.batchnorm1 = nn.BatchNorm1d(self.hidden)
        self.batchnorm2 = nn.BatchNorm1d(self.hidden2)

        self.layer_1 = nn.Linear(self.num_feature,  self.hidden)
        self.layer_2 = nn.Linear( self.hidden, self.hidden2)
        self.layer_3 = nn.Linear( self.hidden2, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x= self.dropout(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x= self.dropout(x)

        x = self.layer_3(x)
        #x = self.sigmoid(x)

        return x


    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        return torch.square(w).sum()

class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)


def predict_protein(mutation_features, model, device, protein_name, length =2200):

  # pad to length AA
  N= length
  fl_dif_pad, positions, proteins_temp =[], [], []
  for i, mut in mutation_features.iterrows():
    mut_temp = mut.mutation.split('_')[1]
    proteins_temp.append(mut.mutation.split('_')[0])
    positions.append(mut_temp[1: -1])
    #a = mut['fl_dif']
    a = mut['Log dif']
    new_a = a + [0] * (N - len(a))
    fl_dif_pad.append(new_a)
  mutation_features['fl_dif_pad'] = fl_dif_pad

  stacked_flat_drgn =[]
  for i, mut in mutation_features.iterrows():
    stacked_flat_drgn.append(torch.tensor(mut['fl_dif_pad']))

  stacked_drgn = torch.stack(stacked_flat_drgn)
  print(stacked_drgn.shape)

  labels_drgn = [random.randint(1, 100) for _ in range(len(stacked_drgn))] # needed for cosntruction of loaders
  X_drgn, y_drgn = np.array(stacked_drgn), np.array(labels_drgn)

  drgn_dataset = ClassifierDataset(torch.from_numpy(X_drgn).float(), torch.from_numpy(y_drgn).long())
  drgn_loader = DataLoader(dataset=drgn_dataset, batch_size=1 , drop_last=True)

  y_pred_list = []
  predictions_drgn= []
  model.eval()

  with torch.no_grad():
      for X_batch, _ in drgn_loader:
          X_batch = X_batch.to(device)
          y_test_pred = model(X_batch)
          predictions_drgn.extend(torch.sigmoid(y_test_pred).cpu().detach().numpy().tolist())

  flat_list = []
  for sublist in predictions_drgn:
      for item in sublist:
          flat_list.append(item)

  newList = [round(n, 4) for n in flat_list]

  return newList


"""Confidence score"""
def calculation_WT_MUT(uniprot, all_mutations, msa_path, tokenizer, model, device, m):
  print(uniprot)

  dif_dif_in, mutations_in, log_prob_in, =[], [], []

  ## Read in a sequence alignment from a fasta file
  if os.path.isfile(msa_path + uniprot+ ".a3m"): # True if file exists
    name_msa_file = msa_path + uniprot+ ".a3m"
  else:
    print('MSA not found in folder !')

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

  arr_WT = np.array(test_features_WT)
  seq_temp = torch.tensor(arr_WT)
  arr_WT = m(seq_temp) # use when you want to reduce dimensions from 1024 to 20
  arr_WT =arr_WT.numpy()


  columns = range(0, arr_WT.shape[1])
  differences_WT= []
  for col in columns:
      first_col = arr_WT[:, col]
      gmm = GaussianMixture(n_components=1).fit(first_col)
      densities_temp = gmm.score_samples(first_col)
      threshold_temp = np.percentile(densities_temp, 1)
      differences_WT.append(densities_temp[0] - threshold_temp)


  ### Calculate differences of all mutations of gene
  for k, mut in all_mutations.iterrows():
    differences_MUT = []
    AA_orig = mut['AA_orig'] # For ProteinGym
    AA_targ = mut['AA_targ'] # For ProteinGym

    diction_test = {} # dictionary containing the difference of log-probabilities of mutation from the lof-prob of WT
    mut_seq = mut['mut_sequence']# mutated sequence
    position = int(mut['position']) - 1
    columns = range(0, arr_WT.shape[1])


    for col in columns:
      if col == position:
        first_col = arr_WT[:, col]
        gmm = GaussianMixture(n_components=1).fit(first_col)
        densities_temp = gmm.score_samples(first_col)
        threshold_temp = np.percentile(densities_temp, 1)
        averages_log_prob = round(Average(densities_temp), 3) # include the WT in the log-probability calculation
        break

    #deep copy of WT array
    arr_WT_MUT = arr_WT.copy()

    # Mutant - threshold
    new_str = [str(x) for x in mut_seq]

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
        first_col = arr_WT_MUT[:, col]
        gmm = GaussianMixture(n_components=1).fit(first_col)
        densities_temp = gmm.score_samples(first_col)
        threshold_temp = np.percentile(densities_temp, 1)
        differences_MUT.append(densities_temp[0] - threshold_temp)

    mutant = uniprot+'_'+AA_orig+str(mut['position'])+AA_targ
    mutations_in.append(mutant) # append mutation
    log_prob_in.append(averages_log_prob) # append log-probability of MSA position
    dif_dif_in.append([differences_WT[i] - differences_MUT[i] for i in range(len(differences_MUT))]) # difference of WT and Mutated sequence log probabiities

  return log_prob_in, mutations_in, dif_dif_in


def load_uniprot_fasta(identifier): #loads fasta file for a given UniProt identifier
    link = "http://www.uniprot.org/uniprot/" + identifier + ".fasta"

    str_data = requests.get(link).content.decode('utf-8')
    fasta = str_data.split('>')
    fasta_all=[]
    for seq in fasta[1:]:
      temp = seq.splitlines()[1:]
      temp = ''.join(temp)
      fasta_all.append(temp)
    return fasta_all[0]

def subst_download_new(uniprot, start, end):
        '''
        Input: Uniprots ID,  for 19 other AA substitutions from start to end
        '''
        # Download sequence from uniprot
        sequence = load_uniprot_fasta(uniprot)
        df= substitute(sequence, start, end)

        return df
def substitute(sequence, start, end):
        '''
        Input: Uniprots ID,  for 19 remaining AA substitutions from start to end
        '''
        # if entire sequence, uncomment following 2 lines:
        # start = 1
        # end=len(sequence) +1

        sequence_part = list(sequence[start-1:end-1]) # keep posit_range of sequence - example: 193-280

        AA_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        mut_sequence, AA_targ, AA_orig, position = [], [], [], []


        for i, AA in enumerate(sequence_part):
                mut_seq = list(sequence)
                remaining_AA = AA_list.copy()
                remaining_AA.remove(AA)
                for k in remaining_AA:
                        mut_seq[start+i-1] = k
                        mut_sequence.append(''.join(mut_seq))
                        AA_orig.append(AA)
                        AA_targ.append(k)
                        position.append(start+i)
        d = {'uniprot id': [uniprot]*len(AA_targ), 'WT_sequence' : sequence, 'mut_sequence':mut_sequence, 'AA_orig': AA_orig, 'position' : position, 'AA_targ' : AA_targ}
        df = pd.DataFrame(data =d)
        return df

def find_WT(uniprot, fasta_uniprot_canonical_path, fasta_uniprot_isoform_path):
  # find uniprot WT sequence of protein
  result_dict_canonical = fasta_to_dict(fasta_uniprot_canonical_path)
  result_dict_isoform = fasta_to_dict(fasta_uniprot_isoform_path)
  #print(result_dict_canonical[uniprot])

  try:
    if uniprot in result_dict_canonical:
      WT_sequence = result_dict_canonical[uniprot]

    elif uniprot in result_dict_isoform:
      WT_sequence = result_dict_isoform[uniprot]
    return WT_sequence

  except:
    print('uniprot not in fasta')
    return None

def fasta_to_dict(file_path):
    fasta_dict = {}

    with open(file_path, 'r') as fasta_file:
        current_accession = None
        current_sequence = []

        for line in fasta_file:
            line = line.strip()

            if line.startswith('>'):
                # If a new accession is found, save the previous one (if any)
                if current_accession is not None:
                    fasta_dict[current_accession] = ''.join(current_sequence)

                # Extract the accession from the header line
                current_accession = line.split('|')[1]
                current_sequence = []
            else:
                # Append sequence lines
                current_sequence.append(line)

        # Save the last entry
        if current_accession is not None:
            fasta_dict[current_accession] = ''.join(current_sequence)

    return fasta_dict

def save_fasta_file(uniprot, sequence):
    file_name = f"{uniprot}.fasta"

    with open(file_name, 'w') as fasta_file:
        fasta_file.write(f">{uniprot}\n{sequence}\n")

def dict_to_fasta(ordered_dict):
    fasta_lines = []

    for header, sequence in ordered_dict.items():
        # Format each entry as a FASTA record
        header=header.split('\t')[0]
        fasta_lines.append(f">{header}")
        fasta_lines.append(sequence)

    # Join the lines to create the final FASTA string
    fasta_string = "\n".join(fasta_lines)
    return fasta_string

def save_fasta_to_file(ordered_dict, filename):
    fasta_content = dict_to_fasta(ordered_dict)

    with open(filename, 'w') as fasta_file:
        fasta_file.write(fasta_content)
