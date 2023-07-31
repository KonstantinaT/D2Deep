D2Deep: Combining evolution and protein language models for cancer driver mutation prediction
=========

The method capitalizes on the abundance of available protein sequences, state-of-the-art Large Language Model architectures (Prot-T5 [1]) and Evolutionary Information from Multiple Sequence Alignments. The statistical model incorporated makes possible the calculation of confidence score for each prediction facilitating its interpretation.

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

Training
============



<img src="https://github.com/KonstantinaT/Predictor/assets/22578534/789c2440-a8da-4676-9791-de2c01a3bbed" width="70%" height="50%">

Confidence calculation
============

Inference
============

Website
============
To assess the effect of a mutation and view all comprehensive set of tools for querying and visualizing protein mutations, visit : https://tumorscope.be/d2deep/
Example for TP53 (P04637):

<img src="https://github.com/KonstantinaT/Predictor/assets/22578534/bee303ae-38a0-4049-97dd-ae1b18fa8a88" width="70%" height="70%">

References
=============
[1] Elnaggar A, Heinzinger M, Dallago C, et al. ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE Trans Pattern Anal Mach Intell. 2022;44(10):7112-7127. doi:10.1109/TPAMI.2021.3095381
