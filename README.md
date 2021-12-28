# RUL_Prediction_Benchmarking

Deep Learning in Time Series Forecasting

This repository contains the experiments done in the work "Benchmarking of Regression and Classification Deep Learning Methods in Remaining Useful Life Prediction of Structures Subject to Fatigue" by Akrim Anass, Gogu Christian, Guillebot de Nerville Thomas, Strähle Paul, Waffa Pagou Brondon, Salaün Michel and Vingerhoeds Rob.


# Setup

Reproducible results are possible using the Keras Tensorflow library. This code was tested on Python 3.8.5, Pandas 0.25.1, Ubuntu 18.04, Anaconda 4.7.11, Tensorflow version 2.3.0, and CUDA 11.0. It requires V100 GPUs.

# Description

The idea behind this project is to benchmark on a fatigue damage prognostics problem the deep learning algorithms most commonly used in Time Series Forecasting : Recurrent Neural Network (RNN), Long short-term memory (LSTM), Gated Recurrent Unit (GRU), 1D-Convolutional Neural Network (CNN), and Temporal Convolutional Network
(TCN). The main goals of this project can be summarized as follows : 

◦ Propose a framework and code for synthetically generating datasets, simulating the fatigue crack growth that can be used in PHM, allowing to test the Machine Learning techniques (especially Deep Neural Networks) on large amounts of labeled data.

◦ Investigate different formulations of the RUL estimation problem in terms of regression (pointwise estimates) or classification (estimation bounds).

◦ Implement a robust and reproducible training strategy, allowing to choose the most suitable model for the considered problem with optimal model hyperparameters.

◦ Benchmark several Deep Learning approaches in terms of their performance, investigating in particular the variation of their performance when the amount
of labeled data increases.

The code used to generate datasets is detailed on the Jupyter Notebook ``GenerateSet.ipynb``, and the saved models with the training codes for the methods are available on the ``models`` directory. 



# Acknowledgements

◦ This work was partially funded by Occitanie region under the Predict project. This funding is gratefully acknowledged. 

◦ This work has been carried out on the supercomputers PANDO (ISAE Supaero, Toulouse) and Olympe (CALMIP, Toulouse, project n°21042). Authors are grateful to ISAE Supaero and CALMIP for the hours allocated to this project.
