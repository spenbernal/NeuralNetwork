import numpy as np
import torch
import torch.nn as nn

class VAE:
    def __init__(self, encoder, decoder, latent_dim, reconstruction_loss) -> None:
        # MLP, CNN, RNN, ...
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.reconstruction_loss = reconstruction_loss
        
    def encode(self, X):
        # X: N,D
        # encoder returns mean and log variance for latent posterior
        self.mu, self.logvar = self.encoder.forward(X) 
        self.logvar = np.clip(self.logvar, -20, 20)
        self.var = np.exp(self.logvar)
        # each data point gets its own latent representation
        # N, L
        return
    
    def decode(self, z):
        # z -> X_recon
        return self.decoder.forward(z)
        
    def reparameterize(self):
        self.std = np.sqrt(self.var)
        self.epsilon = np.random.randn(*self.mu.shape)
        self.z = self.mu + self.std * self.epsilon
        return self.z

    def KL_divergence(self):
        # assume latent prior is standard normal 
        # latent posterior has parameters outputted from encoder
        KL = (-1/2) * (self.logvar - self.var - self.mu**2 + 1).sum(axis= 1)
        return KL
    
    def forward(self, X):
        # X -> mu, var -> reparameterize -> X_recon
        self.encode(X)
        z = self.reparameterize()
        X_recon = self.decode(z)
        
        # ELBO
        loss_kl = self.KL_divergence().mean()
        loss_recon = self.reconstruction_loss.forward(X, X_recon)
        loss = loss_recon + loss_kl
        return X_recon, loss, loss_recon, loss_kl
        
    def backwards(self):
        delta_recon = self.reconstruction_loss.backwards()
        delta_z = self.decoder.backwards(delta_recon)
        delta_mu = delta_z.copy()
        delta_logvar = delta_z * 0.5 * self.std * self.epsilon
        
        # KL contribution
        N = self.mu.shape[0]
        delta_mu += self.mu / N
        delta_logvar += 0.5 * (self.var - 1) / N
        
        delta_X = self.encoder.backwards(delta_mu, delta_logvar)
        return delta_X
    
    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()
    
    def update(self, eta):
        self.encoder.update(eta)
        self.decoder.update(eta)
    
    def embed(self, X):
        self.encode(X)
        return self.mu
    
    def generate(self, n):
        # sample from latent prior
        z = np.random.randn(n, self.latent_dim)
        # generate data
        logits = self.decode(z)
        probs = 1 / (1 + np.exp(-logits))
        return probs
    
