#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:29:49 2021
Get data from the online dataset and extract active neurons
get_all_subject: get all data for all subjects
get_ind_active_neurons: get thresholded (active) individual neuron activity
"""

import numpy as np
import os, requests
# get all subjects raw data from the online database
def get_all_subject():
  fname = []
  for j in range(3):
    fname.append('steinmetz_part%d.npz'%j)
  url = ["https://osf.io/agvxh/download"]
  url.append("https://osf.io/uv3mw/download")
  url.append("https://osf.io/ehmw2/download")

  for j in range(len(url)):
    if not os.path.isfile(fname[j]):
      try:
        r = requests.get(url[j])
      except requests.ConnectionError:
        print("!!! Failed to download data !!!")
      else:
        if r.status_code != requests.codes.ok:
          print("!!! Failed to download data !!!")
        else:
          with open(fname[j], "wb") as fid:
            fid.write(r.content)
  #Data loading
  alldat = np.array([])
  for j in range(len(fname)):
    alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))
  return alldat

def get_ind_active_neurons(alldat,subject,brain_region,min_spk_per_trial,plotopt=None):
  # get thresholded individual neuron activity
  # subject: integer number
  # brain_region
  # min_spk_per_trial

  dat = alldat[subject]
  NeuronInd = dat['brain_area']==brain_region
  spk=dat['spks']   # neuron, trial, time
  spk=spk[NeuronInd,:,:]
  resp=dat['response']; # response of each trial
  spk_per_trial=10 # minimum spk per trial
  spk=np.transpose(spk, (2, 1, 0))    # swap the dimension
  active_neuron_ind=spk.sum(axis=(0,1))>=spk_per_trial*spk.shape[1]     # thresholded neuron ind
  act_spk=spk[:,:,active_neuron_ind]     # thresholded active neurons
  print('active spk shape',act_spk.shape) # time, trial, neuron
  
  if plotopt is not None:   # plot before and after sum of spikes per trial
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(spk.sum(axis=(0)))
    plt.colorbar()
    plt.xlabel('Neuron')
    plt.ylabel('Trial')
    plt.title('Silent neurons are silent in all trials')

    plt.subplot(122)
    plt.imshow(act_spk.sum(axis=(0)))
    plt.colorbar()
    plt.xlabel('Neuron')
    plt.ylabel('Trial')
    plt.title('Active neurons')
  return dat,NeuronInd,act_spk,resp # dat: all data for the subject; NeuronInd: neuron index in the specified brain_area; active spk data; response (-1 or 1 or 0) for all trials
