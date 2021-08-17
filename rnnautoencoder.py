import numpy as np
import matplotlib.pyplot as plt
import itertools

from tqdm.notebook import tqdm, trange

import torch
from torch.utils.data import DataLoader
nn = torch.nn

import nmastandard as nmas

class RNNAE(nn.Module):
  def __init__(self, in_dim, latent_dim, enc_lst, dec_lst):
    """
    Initialize AutoEncoder
    ---
    Parameters:
    * in_dim : Number of input dimensions
    * latent_dim : Size of latent space
    * enc_lst : List of number of hidden nodes at each encoder layer
    * dec_lst : List of number of hidden nodes at each decoder layer
    """

    super(RNNAE, self).__init__()

    self.in_dim = in_dim
    self.out_dim = in_dim

    # Create Encoder Model
    layers_a = [[nn.GRU(in_dim, enc_lst[0], bias=True),]]
    layers_a += [[nn.GRU(enc_lst[idim], enc_lst[idim+1], bias=True)] for idim in range(len(enc_lst)-1)]
    layers_a += [[nn.GRU(enc_lst[-1], latent_dim, bias=True)]]
    enc_layers = []
    for layer in layers_a:
      enc_layers += layer
    self.enc_model = enc_layers #nn.Sequential(*enc_layers)


    # Create Decoder Model
    layers_a = [[nn.GRU(latent_dim, dec_lst[0], bias=True)]]
    layers_a += [[nn.GRU(dec_lst[idim], dec_lst[idim+1], bias=True)] for idim in range(len(dec_lst)-1)]
    layers_a += [[nn.Linear(dec_lst[-1], in_dim, bias=True), nn.ReLU()]]
    dec_layers = []
    for layer in layers_a:
      dec_layers += layer
    self.dec_model = dec_layers #nn.Sequential(*dec_layers)

    self.params = nn.ParameterList()
    for layer in self.enc_model+self.dec_model:
      self.params.extend(layer.parameters())
    
  def custom_seq(self, model, x):
    for layer in model:
      try:
        x, _ = layer(x)
      except:
        x = layer(x)
    return x

  def encode(self, x):
    '''
    Enocdes x into the latent space
    ---
    Parameters:
    * x (torch.tensor) : The dataset to encode (size: num_examples x in_dim)

    Returns:
    * l (torch.tensor) : Projection into the latent space of original data (size: num_examples x latent_dim)
    '''
    return self.custom_seq(self.enc_model, x)

  def decode(self, l):
    '''
    Decode l from the latent space into the initial dataset
    ---
    Parameters:
    * l (torch.tensor) : The encoded latent space representation (size: num_examples x latent_dim)

    Returns:
    * x (torch.tensor) : Approximation of the original dataset encoded (size: num_examples x in_dim)
    '''
    return self.custom_seq(self.dec_model, l)

  def forward(self, x):
    '''
    Feed raw dataset through encoder -> decoder model in order to generate overall approximation from latent space
    ---
    Parameters:
    * x (torch.tensor) : The dataset to encode (size: num_examples x in_dim)

    Returns:
    * x (torch.tensor) : Approximation of the original dataset from the encoded latent space (size: num_examples x in_dim)
    '''
    flat_x = x
    h = self.encode(flat_x)
    return self.decode(h).view(x.size())

def train_autoencoder(autoencoder, dataset, device, val_dataset=None, epochs=20, batch_size=250,
                      seed=0):
  '''
  Train the provided "autoencoder" model on the provided tensor dataset.
  ---
  Parameters:
  * autoencoder (AE) : AE model to train
  * dataset (torch.tensor) : The dataset to encode (size: num_examples x in_dim)
  * device (str) : Device to use for training ('cuda' or 'cpu')
  * val_dataset (torch.tensor) : The datset to encode for validation loss (size: num_examples x in_dim)
  * epochs (int) : Number of iterations through the entire dataset on which to train
  * batch_size (int) : Number of examples in randomly sampled batches to pass through the model
  * seed (int) : Random seed to use for the model

  Returns:
  * mse_loss (torch.tensor) : List of Mean Squared Error losses by training timestep
  '''

  autoencoder.to(device)
  optim = torch.optim.Adam(autoencoder.params,
                           lr=1e-3,
                           #weight_decay=1e-5
                           )
  loss_fn = nn.MSELoss()
  # loss_fn = nn.PoissonNLLLoss(log_input=True)

  g_seed = torch.Generator()
  g_seed.manual_seed(seed)
  loader = DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      pin_memory=True,
                      num_workers=2,
                      worker_init_fn=nmas.seed_worker,
                      generator=g_seed)
  

  mse_loss = torch.zeros(epochs * len(dataset) // batch_size, device=device)
  
  # Creating a version of losses for tracking full dataset loss
  full_mse_loss = torch.zeros(epochs, device=device)
  full_acc = torch.zeros(epochs, device=device)
  if val_dataset is not None:
    full_val_loss = torch.zeros(epochs, device=device)
    full_val_acc = torch.zeros(epochs, device=device)
  else:
    full_val_loss = None
    full_val_acc = None
  
  i = 0
  for epoch in trange(epochs, desc='Epoch'):
    
    
    # Calculate full dataset losses at the end of each epoch
    with torch.no_grad():
      fim_batch = dataset.to(device)
      freconstruction = autoencoder(fim_batch)
      floss = loss_fn(freconstruction.view(fim_batch.size(0), -1),
                      target=fim_batch.view(fim_batch.size(0), -1))
      full_mse_loss[epoch] = floss.detach()
      
      full_acc[epoch] = torch.mean(((freconstruction - fim_batch).abs() < 0.1).float()).detach()
      
      if val_dataset is not None:
          val_im_batch = val_dataset.to(device)
          val_reconstruction = autoencoder(val_im_batch)
          val_loss = loss_fn(val_reconstruction.view(val_im_batch.size(0), -1),
                            target=val_im_batch.view(val_im_batch.size(0), -1))
          full_val_loss[epoch] = val_loss.detach()

          full_val_acc[epoch] = torch.mean(((val_reconstruction - val_im_batch).abs() < 0.1).float())
    
      if epoch % 10 == 0:
        print(f'MSE Train @ {epoch}: Loss — {full_mse_loss[epoch].cpu()}, Acc — {full_acc[epoch].cpu()}')
        if val_dataset is not None:
          print(f'\tVal @ {epoch}: Loss — {full_val_loss[epoch].cpu()}, Acc — {full_val_acc[epoch].cpu()}')
    

    # print(len(list(itertools.islice(loader, 1))))
    for im_batch in loader:
      im_batch = im_batch.permute((1,0,2))

      im_batch = im_batch.to(device)
      optim.zero_grad()
      reconstruction = autoencoder(im_batch)
      # write the loss calculation
      loss = loss_fn(reconstruction,
                    target=im_batch)
      loss.backward()
      optim.step()

      # mse_loss[i] = loss.detach()
      # i += 1


  # After training completes, make sure the model is on CPU so we can easily
  # do more visualizations and demos.
  autoencoder.to('cpu')

  tr_mse = full_mse_loss.cpu()
  val_mse = full_val_loss.cpu() if val_dataset is not None else None

  return tr_mse, val_mse

if __name__ == '__main__':
    SEED = 2021
    nmas.set_seed(seed=SEED)
    DEVICE = nmas.set_device()

    
    x_a = np.random.choice(10000, size=10000)
    tmp = np.tile(np.arange(-1,2), (x_a.shape[0],1))
    x = np.tile(x_a.reshape(-1,1), [1, 3]) + tmp

    inx = np.random.choice(x.shape[0])
    
    x = x[inx]
    x = torch.tensor(x).float()

    x_tr = x[:-x.size(0)//5]
    x_val = x[-x.size(0)//5:]

    vae = RNNAE(x.size(-1), 1, [5], [5])
    loss, val_loss = train_autoencoder(vae, x_tr, DEVICE, x_val, epochs=20, batch_size=250, seed=0)

    print(f'Final Training Loss: {loss}')
    print(f'Final Validation Loss: {val_loss}')