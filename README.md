D2Deep: Combining evolution and protein language models for cancer driver mutation prediction
=========

The method capitalizes on the abundance of available protein sequences, state-of-the-art Large Language Model architectures (Prot-T5[1]) and Evolutionary Information from Multiple Sequence Alignments[2]. The statistical model incorporated makes possible the calculation of confidence score for each prediction facilitating its interpretation.
Our website provides a comprehensive set of tools for querying and visualizing protein mutations. 

Software and hardware requirements: The Prot-T5 pre-trained model should be downloaded from: https://huggingface.co/Rostlab/prot_t5_xl_uniref50 before the calculation of features. A GPU with 40GB of RAM is required.


Table of contents
=================

<!--ts-->
   * [Datasets and data processing](https://github.com/KonstantinaT/Predictor/edit/main/README.md#data-acquisition-and-processing)
   * [Training](https://github.com/KonstantinaT/Predictor/edit/main/README.md#training)
   * [Confidence calculation](https://github.com/KonstantinaT/Predictor/edit/main/README.md#confidence-calculation)
   * [Inference](https://github.com/KonstantinaT/Predictor/edit/main/README.md#inference)
   * [Website](https://github.com/KonstantinaT/Predictor/edit/main/README.md#website)
   * [References](https://github.com/KonstantinaT/Predictor/edit/main/README.md#references)
<!--te-->

Datasets and data processing
============

Datasets containing training data, test sets and identifiers mapping can be downloaded through Zenodo: 

The scripts for data curation and processing can be found in *Data_curation/*. The GMM feature extraction code can be found in: *GMM_diff_maxpool.py*

Training
============


<img src="https://github.com/KonstantinaT/Predictor/assets/22578534/789c2440-a8da-4676-9791-de2c01a3bbed" width="70%" height="50%">

Code: *Model.ipynb*


Confidence calculation
============

Code: *all_mutations_oncogene_confidenceB.py*, *confidence.ipynb*

Inference
============

Code: *Inference.ipynb*


Website
============
To assess the effect of a mutation and view all comprehensive set of tools for querying and visualizing protein mutations, visit : https://tumorscope.be/d2deep/

Example for TP53 (P04637):

<img src="https://github.com/KonstantinaT/Predictor/assets/22578534/bee303ae-38a0-4049-97dd-ae1b18fa8a88" width="70%" height="70%">

References
=============
[1] Elnaggar A, Heinzinger M, Dallago C, et al. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Trans Pattern Anal Mach Intell. 2022;44(10):7112-7127. doi:10.1109/TPAMI.2021.3095381

[2] Steinegger, M., Söding, J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat Biotechnol 35, 1026–1028 (2017). https://doi.org/10.1038/nbt.3988
