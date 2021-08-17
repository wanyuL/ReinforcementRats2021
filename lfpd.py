import torch
from torch.utils.data import Dataset
import numpy as np
import nitime as nit
import os, urllib.request
from sklearn.preprocessing import StandardScaler

def mtspectogram(data,winsize,fs,step=1,uselog=True):
    """ Computes multi-taper spectogram for data"""
    padsize = (round(winsize-1/2))
    padded = np.pad(data,padsize,mode='reflect')
    psds = []
    ix = 1
    while ix <= len(data):
        f, psd_mt, nu = nit.algorithms.multi_taper_psd(
                padded[(padsize+ix):(winsize+padsize+ix)], Fs=fs, adaptive=True, jackknife=False
            )
        if uselog:
            psd_mt = np.log(psd_mt)
        psds.append(psd_mt)
        ix += step
    return (f,psds)


class LFPDataset(Dataset):
    """ Dataset with loader for PSD of Steinmetz LFP data, including multitaper 
    power spectrum analysis
    Data: trials x bands x bins"""
    def __init__(self,winsize=51,fs=100,spectralbands=[],subject=11) -> None:
        file = 'steinmetz_lfp.npz'
        if not os.path.exists(file):
            urllib.request.urlretrieve('https://osf.io/kx3v9/download',filename=file)
        alllfpdat = np.load(file, allow_pickle=True)['dat']
        # only use date for one subject
        self.lfpdat = alllfpdat[subject]
        self.winsize = winsize
        self.fs = fs
        if len(spectralbands) == 0:
            # default bands
            self.spectralbands = [(0,10),(10,27),(27,50)]
        else:
            self.spectralbands = spectralbands
        self.subject = subject

        print('Loaded: {}'.format(file))
    
        #Calculate spectral band power in the defined bands for one subject
        # Size area 
        # trials x bands x bins
        #lfpareas = self.lfpdat['brain_area_lfp']
        lfpareas = ['MOs']
        ntrials = self.lfpdat['lfp'].shape[1]
        nbands = len(self.spectralbands)
        self.bandpsds = {}
        for area in lfpareas:
            areabands = []
            areaindex = lfpareas.index(area)
            for trial in range(ntrials):
                triallfp = self.lfpdat['lfp'][areaindex][trial]
                freqs,specgram = mtspectogram(triallfp,self.winsize,self.fs)
                specgram = np.array(specgram)
                bandspecgram = []
                for band in range(nbands):
                    llim,hlim = self.spectralbands[band]
                    bandspecgram.append(np.mean(specgram[:,(freqs > llim) & (freqs <= hlim)],axis=1))
                areabands.append(np.array(bandspecgram))
            self.bandpsds[area] = np.array(areabands)
        scaler = StandardScaler()
        nparr = np.array(self.bandpsds['MOs'])
        nparr = scaler.fit_transform(nparr.reshape(-1, nparr.shape[-1])).reshape(nparr.shape)
        self.samples = torch.from_numpy(nparr).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx].flatten()
