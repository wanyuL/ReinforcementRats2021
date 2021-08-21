# Low Dimensional Embedding of Neural Responses
NMA-DL (2021) group project: Low Dimensional Embedding of Neural Responses

In this repository, we construct several artifitial neural networks (ANNs) using **Pytorch**. The goal is to use these seqeunce-to-seqeunce models to extract low-dimensional latent variables from the population Neuropixel data from [Steinmetz et al. 2019 Natrue](https://www.nature.com/articles/s41586-019-1787-x).

### ANNs:
1. Autoencoder: `autoencoder.py`
2. Simple GRU: encoder=GRU, decoder=Linear layer; `rnn0.py`
3. GRU based autoencoder: encoder=GRU, decoder=GRU + Linear layer(final layer); `rnnautoencoder.py`

### Data loading:
1. `gen_fake_data.py` generates fake Poisson neuronal data, which could be used for initial test of ANNs.
2. `get_active_neurons.py` could load the raw spiking data and extract active neurons. (We found many silent neurons in the dataset).
3. `lfpd.py` extracts LFP data.


### Results:
In this project, we check the neural data **reconstruction errors** (MSE) of different state-of-art ANNs and traditional PCA (results could be found in the second part of `RNN_Notebook.ipynb`). We then do logistic regression on extracted latent varibales (independent variables) and behavorial outputs (dependent variable) of each trial and check the **classification error** of each methods. Classification related code could be found in `LOO_CV.ipynb` and `BehavioralClassifier.ipynb`.

We also visualize the **latent trajectories** and their **behavioral representational similarity matrix** (RSM), which could be found in the first part of `RNN_Notebook.ipynb`.

* Note: all the code and results are preliminary and unpublished. If you have any questions, please contact the contributors. 



