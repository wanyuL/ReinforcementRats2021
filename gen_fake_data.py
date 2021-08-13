#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 19:40:14 2021
generate fake data; adapted from NMA-DL project tutorial
"""
import numpy as np

def gen_fake_spk(NT,ntrials,NN,ncomp): 
   # generate fake numpy neural firing rate data
   # NT: number of time bins per trial per neuron; ntrials: number of trials ; NN: # number of neurons; ncomp: number of latents

  np.random.seed(42)    # set seed
  # this is the recurrent dynamics matrix, which we made diagonal for simplicity
  # values have to be smaller than 1 for stability
  A0 =  np.diag(.8 + .2 * np.random.rand(ncomp,))
  C0 = .025 * np.random.randn(ncomp, NN)
  # start by initializing the latents
  y       = 2 * np.random.randn(ncomp)
  latents = np.zeros((NT, ntrials, ncomp))
  # we run the dynamics forward and add noise (or "innovations") at each timestep
  for t in range(NT):
    y = y @ A0 +  np.random.randn(ntrials, ncomp)
    latents[t] = y

  # we now project the latents to the neuron space and threshold to generate firing rates
  fake_spk_rates = np.maximum(0, latents @ C0)
  # now we draw poisson counts to simulate the number of spikes a neuron fires randomly
  fake_spk = np.random.poisson(fake_spk_rates)
  return fake_spk, fake_spk_rates, latents # fake_spk, fake_spk_rates dimention: #time,#trial,#neuron ; latents dimension: #time,#trial,#latent