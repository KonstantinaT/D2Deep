import pandas as pd
import re
import os
import shutil
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import gc
import requests
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import wilcoxon
from collections import OrderedDict, Counter
from csv import DictWriter
from sklearn.mixture import GaussianMixture
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
from sklearn.neighbors import KernelDensity
from evcouplings.align import Alignment, map_matrix, read_fasta
from D2D_functions_colab import *


#import Pre-Trained model
tokenizer = T5Tokenizer.from_pretrained("/scratch/brussel/104/vsc10400/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("/scratch/brussel/104/vsc10400/prot_t5_xl_uniref50")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()

#import D2Deep model
h = 4096
hidden2=2048

D2Deep_model = Classifier2L(h, hidden2, 0.3).to(device)
D2Deep_model.load_state_dict(torch.load('model'))
D2Deep_model.eval()

m = nn.MaxPool1d(50) # Max Pooling for reduction of features from 1024 to 50 per AA
curwd = os.getcwd()
msa_path= str(curwd) + '/all_msas/'

protein_list = ['Q9BZD2', 'O00141', 'P31040', 'Q01196-8', 'Q68DV7', 'P04049', 'Q92698', 'Q06124-2', 'P10644', 'P14222']

for uniprot in protein_list:
    all_mutations = pd.read_csv(uniprot+'_all.csv')

    #Calculate GMM features and confidence log_prob
    log_prob_temp, mutations, dif_dif = calculation_WT_MUT(uniprot, all_mutations, msa_path, tokenizer, model, device, m)

    confidence_df = pd.DataFrame(list(zip(mutations, dif_dif, log_prob_temp)), columns = ['mutation', 'Log dif', 'Log_prob'])
    diction_test = confidence_df.to_dict()

    # D2Deep predictions
    predictions = predict_protein(confidence_df, D2Deep_model, device, uniprot)
    confidence_df['D2Deep_prediction'] = predictions

    # final confidence calculation and AF2 addition
    confidence_df['uniprot id'] = confidence_df.mutation.str.split(pat='_',expand=True)[0]
    confidence_df['conc_mutation'] = confidence_df.mutation.str.split(pat='_',expand=True)[1]
    confidence_df['AF2_name'] = ['AF-'+uniprot+'-F1-model_v4'] * len(confidence_df)

    final_df = normalise_confidence(confidence_df)
    final_df.to_csv(uniprot+'_d2d_results_confidence.csv')
