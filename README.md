D2Deep: Combining evolution and protein language models for cancer driver mutation prediction
=========

The method capitalizes on the abundance of available protein sequences, state-of-the-art Large Language Model architectures (Prot-T5[1]) and Evolutionary Information from Multiple Sequence Alignments[2]. The statistical model incorporated makes possible the calculation of confidence score for each prediction facilitating its interpretation.
Our website provides a comprehensive set of tools for querying and visualizing protein mutations. 

Software and hardware requirements: The Prot-T5 pre-trained model should be downloaded from: https://huggingface.co/Rostlab/prot_t5_xl_uniref50 before the calculation of features. A GPU with 40GB of RAM is required.


Table of contents
=================

<!--ts-->
   * [Datasets and data processing](https://github.com/KonstantinaT/D2Deep/blob/main/README.md#datasets-and-data-processing)
   * [Training](https://github.com/KonstantinaT/D2Deep#training)
   * [D2D feature extraction: single and multiple mutations](https://github.com/KonstantinaT/D2Deep/blob/main/README.md#example-of-d2d-feature-extraction-for-single-and-multiple-mutations)
   * [Example of inference and confidence calculation](https://github.com/KonstantinaT/D2Deep/blob/main/README.md#example-of-inference-and-confidence-calculation) 
   * [Website](https://github.com/KonstantinaT/D2Deep#website)
   * [References](https://github.com/KonstantinaT/D2Deep#references)
<!--te-->

Datasets and data processing
============

Datasets containing predictions, training data, test sets and identifiers mapping can be downloaded through Zenodo: https://zenodo.org/doi/10.5281/zenodo.8200795

The scripts for data curation and processing can be found in *Data_curation/*.

Training
============


<img src="pipeline.png" width="90%" height="80%">

    Code: Model.ipynb


Example of D2D feature extraction for single and multiple mutations 
============
    Colab: D2D_features_single_multiple.ipynb


Example of inference and confidence calculation 
============
    Colab: Inference.ipynb

 OR

    Python scripts: D2Deep.py, D2Deep_functions.py

Website
============
To assess the effect of a mutation and view all comprehensive set of tools for querying and visualizing protein mutations, visit : https://tumorscope.be/d2deep/

Example for TP53 (P04637):

<img src="webserver.PNG" width="90%" height="80%">

References
=============
[1] Elnaggar A, Heinzinger M, Dallago C, et al. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Trans Pattern Anal Mach Intell. 2022;44(10):7112-7127. doi:10.1109/TPAMI.2021.3095381

[2] Steinegger, M., Söding, J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nat Biotechnol 35, 1026–1028 (2017). https://doi.org/10.1038/nbt.3988
