import numpy as np
import matplotlib.pyplot as plt
import itertools

from tqdm.notebook import tqdm, trange

import torch
from torch.utils.data import DataLoader
nn = torch.nn

import nmastandard as nmas

class AE(nn.Module):
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

    super(AE, self).__init__()

    self.in_dim = in_dim
    self.out_dim = in_dim

    # Create Encoder Model
    layers_a = [[nn.Linear(in_dim, enc_lst[0], bias=True), nn.ReLU()]]
    layers_a += [[nn.Linear(enc_lst[idim], enc_lst[idim+1], bias=True), nn.ReLU()] for idim in range(len(enc_lst)-1)]
    layers_a += [[nn.Linear(enc_lst[-1], latent_dim, bias=True)]]

    enc_layers = []
    for layer in layers_a:
      enc_layers += layer

    self.enc_model = nn.Sequential(*enc_layers)


    # Create Decoder Model
    layers_a = [[nn.Linear(latent_dim, dec_lst[0], bias=True), nn.ReLU()]]
    layers_a += [[nn.Linear(dec_lst[idim], dec_lst[idim+1], bias=True), nn.ReLU()] for idim in range(len(dec_lst)-1)]
    layers_a += [[nn.Linear(dec_lst[-1], in_dim, bias=True)]]

    dec_layers = []
    for layer in layers_a:
      dec_layers += layer

    self.dec_model = nn.Sequential(*dec_layers)

  def encode(self, x):
    return self.enc_model(x)

  def decode(self, x):
    return self.dec_model(x)

  def forward(self, x):
    flat_x = x.view(x.size(0), -1)
    h = self.encode(flat_x)
    return self.decode(h).view(x.size())

def train_autoencoder(autoencoder, dataset, device, epochs=20, batch_size=250,
                      seed=0):
  autoencoder.to(DEVICE)
  optim = torch.optim.Adam(autoencoder.parameters(),
                           lr=1e-2,
                           #weight_decay=1e-5
                           )
  loss_fn = nn.MSELoss()
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
  i = 0
  for epoch in trange(epochs, desc='Epoch'):
    # print(len(list(itertools.islice(loader, 1))))
    for im_batch in loader:
      im_batch = im_batch.to(device)
      optim.zero_grad()
      reconstruction = autoencoder(im_batch)
      # write the loss calculation
      loss = loss_fn(reconstruction.view(batch_size, -1),
                    target=im_batch.view(batch_size, -1))
      loss.backward()
      optim.step()

      mse_loss[i] = loss.detach()
      i += 1
    
    if epoch % 100 == 0:
      print(mse_loss[i])
  
  # After training completes, make sure the model is on CPU so we can easily
  # do more visualizations and demos.
  autoencoder.to('cpu')
  return mse_loss.cpu()

if __name__ == '__main__':
    SEED = 2021
    nmas.set_seed(seed=SEED)
    DEVICE = nmas.set_device()


    x_a = torch.tensor(np.random.choice(10000, size=100000)).float()
    tmp = torch.tensor(np.tile(np.arange(-1,2), (x_a.size(0),1)))
    x = torch.tile(x_a.view(-1,1), [1, 3]) + tmp


    vae = AE(x.size(-1), 1, [5], [5])
    loss = train_autoencoder(vae, x, DEVICE, epochs=100, batch_size=250, seed=0)

    plt.plot(loss)