import pandas as pd
import re
import Bio
import os
import shutil
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import gc
from Bio import SeqIO
from Bio.Seq import Seq
import requests
from utils import *

# download sequences for humsavar and common variants
humsavar = pd.read_csv('humsavar_1221_neut.csv')
humsavar = humsavar.reset_index(drop=True)

humsavar["AA_orig"] = humsavar["Mutation"].str[2:5].apply(AAconvert)
humsavar["AA_targ"] = humsavar["Mutation"].str[-3:].apply(AAconvert)
humsavar["position"] = humsavar["Mutation"].str[5:-3]

transcript_not_found, delins, unknown_genes, count_test = 0, 0, 0, 0
position, AA_orig, AA_targ, sequence, indices_to_excl, canonical, isoform_uniprot, wt_sequence = [], [], [], [], [], [], [],[]
count_found, count_not_found = 0, 0
for i, mut in humsavar.iterrows():

    isoforms, all_transcripts = parse_fasta_url(mut['Protein'])
    iso_names = sorted(isoforms, key=lambda x: int(x.split("-")[-1]) if "-" in x else 0)
    index_isoform = [i for i, v in sorted(enumerate(isoforms), key=lambda x: int(x[1].split("-")[-1]) if "-" in x[1] else 0)]
    all_transcripts = [all_transcripts[i] for i in index_isoform]

    count = 1 # track transcripts
    for isoform, transcript in enumerate(all_transcripts):

      if int(mut['position'])  <= len(transcript):

          if transcript[int(mut['position'])- 1] == mut['AA_orig']:

            isoform_name = iso_names[isoform]  
          
            isoform_uniprot.append(isoform_name)
            mut_seq = changeAAnew(mut['Protein'], mut['AA_orig'], mut['AA_targ'], int(mut['position']), transcript)
            if mut_seq is None:
              print('empty')
              continue

            position.append(int(mut['position']))
            wt_sequence.append(transcript)
            AA_orig.append(mut['AA_orig'])
            count_found +=1
            AA_targ.append(mut['AA_targ'])
            found = True
            sequence.append(mut_seq) 
            break
          else:
            if len(all_transcripts) == count:
              transcript_not_found +=1
              indices_to_excl.append(i)
            count+=1
      else:
          if len(all_transcripts) == count:
            transcript_not_found +=1
            indices_to_excl.append(i)
          count+=1

humsavar_mutations = pd.DataFrame(list(zip(isoform_uniprot, wt_sequence, sequence, AA_orig, position, AA_targ)),
               columns =['uniprot', 'WT_sequence', 'mut_sequence', 'AA_orig', 'position', 'AA_targ'])
humsavar_mutations['label'] = [0]*len(isoform_uniprot)

# save mutations 

humsavar_mutations.to_csv('humsavar_benign_mutations.csv', index=False)
