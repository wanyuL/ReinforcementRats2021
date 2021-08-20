#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 22:08:12 2021
pickle data helper

"""

def save_pkldata(data,filename):
    # filename: 'example_name'
  file = open(f"{filename}.pkl",'wb')
  pickle.dump(data,file)
  return

def load_pkldata(filename):
  file = open(f"{filename}.pkl",'rb')
  data = pickle.load(file)
  return data
  