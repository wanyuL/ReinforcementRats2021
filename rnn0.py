#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN 
adapted from NMA-DL project website
"""
# Imports
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

## input to this model: 

class rnn0(nn.Module):
  def __init__(self, ncomp, NN1, NN2, bidi=False):
    super(rnn0, self).__init__()

    # play with some of the options in the RNN!

    self.rnn = nn.RNN(NN1, ncomp, num_layers = 1, dropout = 0,
                      bidirectional = bidi, nonlinearity = 'tanh')    # relu doesn't work
    # self.rnn = nn.GRU(NN1, ncomp, num_layers = 1, dropout = 0,
    #                   bidirectional = bidi)      # Elman rnn seems to work better than gru and lstm
    self.fc = nn.Linear(ncomp, NN2)

  def forward(self, x):

    y,h = self.rnn(x)    # output, hidden state; we don't need h here; 

    if self.rnn.bidirectional:
      # if the rnn is bidirectional, it concatenates the activations from the forward and backward pass
      # we want to add them instead, so as to enforce the latents to match between the forward and backward pass
      q = (y[:, :, :ncomp] + y[:, :, ncomp:])/2
    else:
      q = y

    # the softplus function is just like a relu but it's smoothed out so we can't predict 0
    # if we predict 0 and there was a spike, that's an instant Inf in the Poisson log-likelihood which leads to failure
    z = F.softplus(self.fc(q), 10)

    return z, q



  def train(rnn0net,train_inputdata,train_matchdata,nepoch,lr,device):
    # rnn0net: rnn model; rnn0(ncomp, NN1, NN2, bidi = True).to(device)
    # nepoch: number of iteration in training loop
    # lr: learning rate; try 0.05 
    # train data dimension: number of time bins,ntrials,number of neurons
    
    # we define the Poisson log-likelihood loss; this is equivalent to nn.PoissonNLLLoss()
    def Poisson_loss(lam, spk):    # lam: lambda of Poisson distribution = output seq of rnn0 ; spk: output trace (output we are trying to fit to)
      return lam - spk * torch.log(lam)


    optimizer = torch.optim.Adam(rnn0net.parameters(), lr)      
    losst = torch.zeros(nepoch, device=device)   # loss over time holder
    for k in range(nepoch):
    # the network outputs the single-neuron prediction and the latents
      
      prd, latv = rnn0net(train_inputdata)     # prediction and latent variables
     
      # our log-likelihood cost
      loss = Poisson_loss(prd, train_matchdata).mean()
      losst[k] = loss
      # train the network as usual
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()


      if k % 100 == 0:
        print(f'iteration {k}, loss {loss.item():.4f}')
    return losst, prd, latv