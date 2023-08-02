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


cmc_all = pd.read_csv("cmc_export.csv",  delimiter = '\t', usecols = ['GENE_NAME', 'ACCESSION_NUMBER', 'Mutation AA', 'Mutation Description AA', 'MUTATION_SIGNIFICANCE_TIER']) # COSMIC: cmc Tier 1,2,3, other mutations

# Filter out silent / nonsense mutations
missense = cmc_all[(cmc_all['Mutation Description AA'] == 'Substitution - Missense')]

# creation of Tier1-2-3 mutations
Tier1_2_3_missense = missense[(missense['MUTATION_SIGNIFICANCE_TIER'] == '1') \
                              | (missense['MUTATION_SIGNIFICANCE_TIER']== '2') \
                                       | (missense['MUTATION_SIGNIFICANCE_TIER']== '3')]
# Exclude delins
Tier1_2_3_missense = Tier1_2_3_missense.reset_index(drop=True)
delins=0
indices_to_excl = []
for i, mut in Tier1_2_3_missense.iterrows():
    string = mut['Mutation AA']
    if 'delins' in string:
      delins+=1
      indices_to_excl.append(i)

Tier1_2_3_missense = Tier1_2_3_missense.drop(Tier1_2_3_missense.index[indices_to_excl]) 


# Read transcript consensus (glioma_All_cosmic.ipynb)
with open("transcript_sequences.csv", 'rb') as fp:
    gene_sequence = pickle.load(fp)

#gene_protein_mapping = pd.read_csv("/content/drive/MyDrive/my_colab/3rdYear/datasets/gene_synonyms_uniprot.txt", sep = '\t')
gene_protein_mapping = pd.read_csv("gene_uniprot2023.txt", sep = '\t') # new version
# Synonymous genes
gene_synonym_df = gene_protein_mapping[['UniProtKB/Swiss-Prot ID', 'Gene Synonym']]
gene_synonym_df.dropna(inplace=True)
gene_synonym_df =gene_synonym_df.drop_duplicates()
gene_synonym_dict = gene_synonym_df.groupby('Gene Synonym')['UniProtKB/Swiss-Prot ID'].apply(lambda g: g.values.tolist()).to_dict()

gene_df = gene_protein_mapping[['UniProtKB/Swiss-Prot ID', 'Gene name']]
gene_df.dropna(inplace=True)
# dropping ALL duplicate values
gene_df =gene_df.drop_duplicates()
gene_dict = gene_df.groupby('Gene name')['UniProtKB/Swiss-Prot ID'].apply(lambda g: g.values.tolist()).to_dict()

#unreviewed genes
gene_unreviewd_synonym_df = gene_protein_mapping[['UniProtKB/TrEMBL ID', 'Gene Synonym']]
gene_unreviewd_synonym_df.dropna(inplace=True)
gene_unreviewd_synonym_df =gene_unreviewd_synonym_df.drop_duplicates()
gene_unreviewd_synonym_dict = gene_unreviewd_synonym_df.groupby('Gene Synonym')['UniProtKB/TrEMBL ID'].apply(lambda g: g.values.tolist()).to_dict()

gene_unreviewd_df = gene_protein_mapping[['UniProtKB/TrEMBL ID', 'Gene name']]
gene_unreviewd_df.dropna(inplace=True)
# dropping ALL gene_unreviewd_df values
gene_unreviewed_df =gene_unreviewd_df.drop_duplicates()
gene_unreviewed_dict = gene_unreviewed_df.groupby('Gene name')['UniProtKB/TrEMBL ID'].apply(lambda g: g.values.tolist()).to_dict()


gene_dict['C5orf60'] = ['A6NFR6'] # missing from gene_uniprot2023.txt
gene_dict['GDF5OS'] = ['Q5U4N7']
gene_dict['SLCO1B7'] = ['F5H094']
gene_dict['CCDC144NL'] = ['Q6NUI1']
gene_dict['PRR34']= ['Q9NV39']
gene_dict['FAM86C1']= ['Q9NVL1']
gene_dict['C17orf82']= ['Q86X59']
gene_dict['C15orf56']= ['Q8N910']
gene_dict['RPSAP58']= ['A0A8I5KQE6']


Tier1_2_3_missense = Tier1_2_3_missense.reset_index(drop=True)
transcript_not_found, delins, unknown_genes, count_test = 0, 0, 0, 0
position, AA_orig, AA_targ, sequence, indices_to_excl, canonical, isoform_uniprot, wt_sequences = [], [], [], [], [], [], [], []
count_found, count_not_found = 0, 0


for i, mut in Tier1_2_3_missense.iterrows():

      gene_name = mut['GENE_NAME']
      string = mut['Mutation AA']
      pos_temp = int(string[3:-1])


      if gene_name in gene_dict:
        uniprot_canonical_list = gene_dict[gene_name]# we may have more than one corresponding proteins for gene
      elif gene_name in gene_synonym_dict:
        uniprot_canonical_list = gene_synonym_dict[gene_name]
      elif gene_name in gene_unreviewd_synonym_dict:
        uniprot_canonical_list = gene_unreviewd_synonym_dict[gene_name]
      elif gene_name in gene_unreviewed_dict:
        uniprot_canonical_list = gene_unreviewed_dict[gene_name]

      else:
        unknown_genes +=1
        indices_to_excl.append(i)
        continue

      count_uniprot =0 # track isoforms
      found=False # used to stop searching follwing isoforms

      for uniprot_canonical in uniprot_canonical_list:
        count_uniprot+=1
        if found == True:
          break


        # load all transcript from uniprot and compare with the COSMIC transcript
        isoforms, all_transcripts = parse_fasta_url(uniprot_canonical)
        iso_names = sorted(isoforms, key=lambda x: int(x.split("-")[-1]) if "-" in x else 0)
        index_isoform = [i for i, v in sorted(enumerate(isoforms), key=lambda x: int(x[1].split("-")[-1]) if "-" in x[1] else 0)]
        all_transcripts = [all_transcripts[i] for i in index_isoform]

        count = 1 # track transcripts
        for isoform, transcript in enumerate(all_transcripts):

          if pos_temp <= len(transcript):

            if transcript[pos_temp- 1] == string[2]:

              isoform_name = iso_names[isoform]
              mut_seq = changeAAnew(isoform_name,string[2], string[-1], pos_temp, transcript)
              if mut_seq is None:
                print('empty')
                continue

              isoform_uniprot.append(isoform_name)
              position.append(pos_temp)
              AA_orig.append(string[2])
              count_found +=1
              AA_targ.append(string[-1])
              found = True
              sequence.append(mut_seq)
              wt_sequences.append(transcript)
              break
            else:
              if len(all_transcripts) == count and count_uniprot == len(uniprot_canonical_list):
                transcript_not_found +=1
                indices_to_excl.append(i)
              count+=1
          elif len(all_transcripts) == count and count_uniprot == len(uniprot_canonical_list):
              transcript_not_found +=1
              indices_to_excl.append(i)
          count+=1
print(len(isoform_uniprot), len(wt_sequences), len(AA_orig))
tier = pd.DataFrame(list(zip(isoform_uniprot, wt_sequences, sequence, AA_orig, position, AA_targ)),
               columns =['uniprot', 'WT_sequence', 'mut_sequence', 'AA_orig', 'position', 'AA_targ'])
tier['label'] = [1]*len(isoform_uniprot)

# save mutations
tier.to_csv('Tier.csv', index=False)


CGI_oncogenic_mut = pd.read_csv("missense_nonsense_CGI_oncogenic_mut.csv", sep = '\t')
print(len(CGI_oncogenic_mut))
print(CGI_oncogenic_mut.head())


CGI_oncogenic_missense = CGI_oncogenic_mut[~CGI_oncogenic_mut['protein'].str.contains('fs')]
CGI_oncogenic_missense = CGI_oncogenic_missense[~CGI_oncogenic_missense['protein'].str.contains('del')]
CGI_oncogenic_missense = CGI_oncogenic_missense[~CGI_oncogenic_missense['protein'].str.contains('_')]
CGI_oncogenic_missense = CGI_oncogenic_missense[~CGI_oncogenic_missense['protein'].str.contains('dup')]
CGI_oncogenic_missense = CGI_oncogenic_missense[CGI_oncogenic_missense['context'] == 'somatic']


CGI_oncogenic_missense = CGI_oncogenic_missense.reset_index(drop=True)
transcript_not_found, delins, unknown_genes, count_test = 0, 0, 0, 0
position, AA_orig, AA_targ, sequence, indices_to_excl, canonical, isoform_uniprot, wt_sequences = [], [], [], [], [], [], [],[]
count_found, count_not_found = 0, 0
for i, mut in CGI_oncogenic_missense.iterrows():

  gene_name = mut['gene']
  string = mut['protein']    
  pos_temp = int(string[3:-1])

  # exclude nonsense mutations
  if string[-1] == '*':
    indices_to_excl.append(i)
    continue

  if gene_name in gene_dict:
    uniprot_canonical_list = gene_dict[gene_name]# we may have more than one corresponding proteins for gene
  elif gene_name in gene_synonym_dict:
    uniprot_canonical_list = gene_synonym_dict[gene_name] 
  elif gene_name in gene_unreviewd_synonym_dict:
    uniprot_canonical_list = gene_unreviewd_synonym_dict[gene_name] 
  elif gene_name in gene_unreviewed_dict:
    uniprot_canonical_list = gene_unreviewed_dict[gene_name]   
  else:
    unknown_genes +=1
    indices_to_excl.append(i)
    continue

  count_uniprot =0 # track isoforms
  found=False # used to stop searching follwing isoforms

  for uniprot_canonical in uniprot_canonical_list:
    count_uniprot+=1 
    if found == True:
      break

    # load all transcript from uniprot and compare with the COSMIC transcript
    isoforms, all_transcripts = parse_fasta_url(uniprot_canonical)
    iso_names = sorted(isoforms, key=lambda x: int(x.split("-")[-1]) if "-" in x else 0)
    index_isoform = [i for i, v in sorted(enumerate(isoforms), key=lambda x: int(x[1].split("-")[-1]) if "-" in x[1] else 0)]
    all_transcripts = [all_transcripts[i] for i in index_isoform]

    count = 1 # track transcripts
    for isoform, transcript in enumerate(all_transcripts):

      if pos_temp <= len(transcript):

          if transcript[pos_temp - 1] == string[2]:
            isoform_name = iso_names[isoform]  
          
            isoform_uniprot.append(isoform_name)
            mut_seq = changeAAnew(gene_name, string[2], string[-1], pos_temp, transcript)
            if mut_seq is None:
              print('empty')
              continue

            position.append(pos_temp)
            wt_sequences.append(transcript)
            AA_orig.append(string[2])
            count_found +=1
            AA_targ.append(string[-1])
            found = True
            sequence.append(mut_seq) 
            break
          else:
            if len(all_transcripts) == count and count_uniprot == len(uniprot_canonical_list):
              transcript_not_found +=1
              indices_to_excl.append(i)
            count+=1

      else:
          if len(all_transcripts) == count and count_uniprot == len(uniprot_canonical_list):
            transcript_not_found +=1
            indices_to_excl.append(i)
          count+=1



cgi = pd.DataFrame(list(zip(isoform_uniprot, wt_sequences, sequence, AA_orig, position, AA_targ)),
               columns =['uniprot', 'WT_sequence', 'mut_sequence', 'AA_orig', 'position', 'AA_targ'])
cgi['label'] = [1]*len(isoform_uniprot)

# save mutations
cgi.to_csv('cgi.csv', index=False)

