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


path = os.getcwd()
filename = "/variant_summary2023.txt"
clinvar = pd.read_csv(path + filename, sep = '\t', usecols = ['Type', 'Name', 'GeneSymbol', 'ClinicalSignificance', 'OriginSimple', 'ReviewStatus']) #'Assembly', 'ClinSigSimple',

clinvar = clinvar.drop_duplicates()

clinvar_missense_nonsense_silent= clinvar.loc[(clinvar['Type'] == 'Variation') \
                                | (clinvar['Type'] == 'single nucleotide variant') \
                                | (clinvar['Type'] == 'protein only')]

clinvar_missense_nonsense_silent_confirmed = clinvar_missense_nonsense_silent.loc[(clinvar_missense_nonsense_silent['ReviewStatus'] == 'criteria provided, multiple submitters, no conflicts') \
                              | (clinvar_missense_nonsense_silent['ReviewStatus'] == 'criteria provided, single submitter') \
                              | (clinvar_missense_nonsense_silent['ReviewStatus'] == 'practice guideline') \
                              | (clinvar_missense_nonsense_silent['ReviewStatus'] == 'reviewed by expert panel') ]

clinvar_missense_nonsense_silent_confirmed = clinvar_missense_nonsense_silent_confirmed[clinvar_missense_nonsense_silent_confirmed['Name'].str.contains(r'\+') == False]
clinvar_missense_nonsense_silent_confirmed = clinvar_missense_nonsense_silent_confirmed[clinvar_missense_nonsense_silent_confirmed['Name'].str.contains('-') == False]
clinvar_missense_nonsense_silent_confirmed = clinvar_missense_nonsense_silent_confirmed[clinvar_missense_nonsense_silent_confirmed['Name'].str.contains('g') == False] # upstream variants
clinvar_missense_nonsense_silent_confirmed = clinvar_missense_nonsense_silent_confirmed[clinvar_missense_nonsense_silent_confirmed['Name'].str.contains('p.') == True]
clinvar_missense_nonsense_confirmed = clinvar_missense_nonsense_silent_confirmed[clinvar_missense_nonsense_silent_confirmed['Name'].str.contains('=') == False]
clinvar_missense_nonsense_confirmed = clinvar_missense_nonsense_confirmed[clinvar_missense_nonsense_confirmed['Name'].str.contains('\+') == False]
clinvar_missense_nonsense_confirmed = clinvar_missense_nonsense_confirmed[clinvar_missense_nonsense_confirmed['Name'].str.contains('\*') == False]          

benign_deleterious= clinvar_missense_nonsense_confirmed.loc[(clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Pathogenic') \
                              | (clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Pathogenic/Likely pathogenic') \
                              | (clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Benign/Likely benign') \
                              | (clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Benign') \
                              | (clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Likely benign') \
                              | (clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Benign, other') \
                              | (clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Pathogenic/Likely pathogenic, risk factor') \
                              | (clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Likely pathogenic, other') \
                              | (clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Pathogenic, risk factor') \
                              | (clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Pathogenic/Likely pathogenic, drug response') \
                              | (clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Pathogenic, drug response') \
                              | (clinvar_missense_nonsense_confirmed['ClinicalSignificance'] == 'Benign/Likely benign, other') ]

# Rename categories Likely Pathogenic etc to either Pathogenic (1) or Benign (0)
benign_deleterious['ClinicalSignificance'] = benign_deleterious['ClinicalSignificance'].replace(['Pathogenic','Pathogenic/Likely pathogenic', 'Pathogenic, risk factor', 'Likely pathogenic, other', 'Pathogenic/Likely pathogenic, drug response', 'Pathogenic, drug response' ], 1)
benign_deleterious['ClinicalSignificance'] = benign_deleterious['ClinicalSignificance'].replace(['Benign/Likely benign', 'Benign', 'Likely benign', 'Benign/Likely benign, other', 'Benign, other' ], 0)



benign_deleterious = benign_deleterious.reset_index(drop=True)

list_AA_orig, list_AA_mut, list_position, list_refseq = [], [], [], []

for ind, row in benign_deleterious.iterrows():
    temp = row['Name']
    temp = temp.split('(')
    string = temp[2]
    name = temp[0]
    if '=' in string:
      string = string[2:-1]
      position = string[3:-1]
      aa_orig = string[:3]
      aa_mut = aa_orig
    else:
      string = string[2:-1]
      position = string[3:-3]
      aa_orig = string[:3]
      aa_mut = string[-3:]
    list_AA_orig.append(AAconvert(aa_orig))
    list_AA_mut.append(AAconvert(aa_mut))
    list_position.append(position)
    list_refseq.append(name)

benign_deleterious['AA_orig'] = list_AA_orig
benign_deleterious['AA_targ'] = list_AA_mut
benign_deleterious['position'] = list_position
benign_deleterious['refseq'] = list_refseq

benign_deleterious_missense = benign_deleterious[benign_deleterious['AA_targ'] != '*']



mart = pd.read_csv("mart_refeqRNA_trembl_swiss.txt", '\t')

mart =  mart.drop_duplicates(subset=['RefSeq mRNA ID','UniProtKB/TrEMBL ID','UniProtKB/Swiss-Prot ID' ], keep='first')
mart = mart.dropna(subset=['UniProtKB/TrEMBL ID', 'UniProtKB/Swiss-Prot ID'], how = 'all')

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


## Find uniprot id (isoform)

# 1) search on mart_export RefSeq mRNA ID --> UniProtKB/Swiss-Prot ID, if not found:
# 2) gene -- > uniprot --> download all transcripts from uniprot website --> test if AA_orig at the right place
benign_deleterious_missense = benign_deleterious_missense.reset_index(drop=True)
transcript_not_found, delins, unknown_genes, count_test = 0, 0, 0, 0
sequence, indices_to_excl, isoform_uniprot, wt_sequences, pos_list, AA_orig_list, AA_targ_list, label = [], [], [], [], [], [], [], []
count_found, count_not_found = 0, 0
count_t = 0
for i, mut in benign_deleterious_missense.iterrows():

  #try:
  refseq = mut['refseq'].split('.')
  refseq = refseq[0]
  mart_temp = mart[mart['RefSeq mRNA ID'] ==refseq]
  AA_orig = mut['AA_orig']
  AA_targ = mut['AA_targ']
  pos_temp = int(mut['position'])

  # 1)


  if len(mart_temp)>0: # refseq symbol found in mart export
    if len(mart_temp) >1:
      count=0
      for k, l in mart_temp.iterrows():
        if notNaN(mart_temp.iloc[count]['UniProtKB/Swiss-Prot ID']):
          uniprot_id = mart_temp.iloc[count]['UniProtKB/Swiss-Prot ID']
        elif notNaN(mart_temp.iloc[count]['UniProtKB/TrEMBL ID']):
          uniprot_id = mart_temp.iloc[count]['UniProtKB/TrEMBL ID']
        count+=1
    else:
      
      if notNaN(mart_temp['UniProtKB/Swiss-Prot ID']).values[0]:
          uniprot_id = mart_temp['UniProtKB/Swiss-Prot ID'].values[0]
      elif notNaN(mart_temp['UniProtKB/TrEMBL ID']).values[0]:
          uniprot_id = mart_temp['UniProtKB/TrEMBL ID'].values[0]

    # load all transcript from uniprot and compare with the COSMIC transcript
    isoforms, all_transcripts = parse_fasta_url(uniprot_id)
    iso_names = sorted(isoforms, key=lambda x: int(x.split("-")[-1]) if "-" in x else 0)
    index_isoform = [i for i, v in sorted(enumerate(isoforms), key=lambda x: int(x[1].split("-")[-1]) if "-" in x[1] else 0)]
    all_transcripts = [all_transcripts[i] for i in index_isoform]

    count = 1 # track transcripts
    for isoform, transcript in enumerate(all_transcripts):
        if int(mut['position'])  <= len(transcript):

          if transcript[int(mut['position'])- 1] == mut['AA_orig']:

            isoform_name = iso_names[isoform]

            isoform_uniprot.append(isoform_name)
            mut_seq = changeAAnew(uniprot_id,AA_orig, AA_targ, pos_temp, transcript)
            if mut_seq is None:
              print('empty')
              continue

            pos_list.append(pos_temp)
            AA_orig_list.append(AA_orig)
            AA_targ_list.append(AA_targ)
            count_found +=1
            sequence.append(mut_seq) 
            label.append(mut['ClinicalSignificance'])
            wt_sequences.append(transcript)

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


  # 2)
  else:

    gene_name = mut['GeneSymbol']

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
            pos_list.append(pos_temp)
            AA_orig_list.append(AA_orig)
            AA_targ_list.append(AA_targ)
            found = True
            sequence.append(mut_seq)
            label.append(mut['ClinicalSignificance'])
            wt_sequences.append(transcript)
            break
          elif len(all_transcripts) == count and count_uniprot == len(uniprot_canonical_list):
              transcript_not_found +=1
              indices_to_excl.append(i)
          count+=1
        elif len(all_transcripts) == count and count_uniprot == len(uniprot_canonical_list):
            transcript_not_found +=1
            indices_to_excl.append(i)

        count+=1

  clinvar_mutations = pd.DataFrame(list(zip(isoform_uniprot, wt_sequences, sequence, AA_orig_list, pos_list, AA_targ_list,label)),
              columns =['uniprot', 'WT_sequence', 'mut_sequence', 'AA_orig', 'position', 'AA_targ', 'label'])


clinvar_mutations.to_csv('clinvar_benign_deleterious_missense.csv', index=False)

count_neg, count_pos = 0,0
for i , mut in clinvar_mutations.iterrows():
  if mut['label'] ==1:
    count_pos +=1
  else:
    count_neg+=1
print(count_neg,count_pos)
