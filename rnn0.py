# Imports
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class rnn0(nn.Module):
  def __init__(self, ncomp, NN1, NN2, dropout=0 ,bidi=False):
    super(rnn0, self).__init__()

    # play with some of the options in the RNN!

    self.rnn = nn.RNN(NN1, ncomp, num_layers = 1, dropout = dropout,
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



  def train(rnn0net,train_inputdata,train_targetdata,nepoch,lr,device,batchsize,lossfn=None,val_inputdata=None,val_targetdata=None):
    # rnn0net: rnn model; rnn0(ncomp, NN1, NN2, bidi = True).to(device)
    # nepoch: number of iteration in training loop
    # lr: learning rate; try 0.05 
    # train data dimension: number of time bins,ntrials,number of neurons
    # lossfn: optional; specified loss function; if None, use default Poission loss; others e.g.: nn.MSELoss()
    # Validation data sets are optional; 

    
    # we define the Poisson log-likelihood loss; 
    def Poisson_loss(lam, spk):    # lam: lambda of Poisson distribution = output seq of rnn0 ; spk: output trace (output we are trying to fit to)
      return lam - spk * torch.log(lam)


    optimizer = torch.optim.Adam(rnn0net.parameters(), lr)      
    train_losst = torch.zeros(nepoch, device=device)   # loss over time holder for training set
    val_losst = torch.tensor(np.nan*np.ones(nepoch), device=device)  # loss over time holder for validation set
      
    # torch.random.seed(10)   # set seed for reproducibility
    for k in range(nepoch):
      ind=torch.randint(0,train_inputdata.shape[1], (batchsize,))   # sample random index for training batch
      train_inputdata_batch=train_inputdata[:,ind,:]
      train_targetdata_batch=train_targetdata[:,ind,:]
    # the network outputs the single-neuron prediction and the latents
      prd, latv = rnn0net(train_inputdata_batch)     # prediction and latent variables
     
      # our log-likelihood cost
      if lossfn is not None:   # no specified loss function: default Poission loss fn
        loss = Poisson_loss(prd, train_targetdata_batch).mean()  # loss for the current batch
      else:
        loss=lossfn(prd, train_targetdata_batch)
      # train the network as usual
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      with torch.no_grad():
        prd_train_full, _ = rnn0net(train_inputdata)
        lossfull_train = Poisson_loss(prd_train_full, train_targetdata).mean()
        train_losst[k] = lossfull_train   # loss for the entire dataset!
        if val_inputdata is not None and val_inputdata is not None:
          prd_val, _ = rnn0net(val_inputdata)
          loss_val = Poisson_loss(prd_val, val_targetdata).mean()
          val_losst[k] = loss_val   # loss for the entire dataset!
        else:
          loss_val=torch.tensor(np.nan,device=device)
      
      if k % 50 == 0:
        print(f'iteration {k}, train_loss {lossfull_train.item():.4f} , val_loss {loss_val.item():.4f}')

    plt.plot(train_losst.detach().cpu().numpy(),label="train_loss_full")
    plt.plot(val_losst.detach().cpu().numpy(),label="val_loss")
    plt.xlabel("nepoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    return train_losst, val_losst, latv  # loss and latent variables
