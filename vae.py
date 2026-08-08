import numpy as np

class VAE:
    def __init__(self, encoder, decoder, reconstruction_loss) -> None:
        # MLP, CNN, RNN, ...
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss = reconstruction_loss
        
    def encode(self, X):
        # X: N,D
        # encoder returns mean and log variance for latent posterior
        self.mu, self.logvar = self.encoder.forward(X) 
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
        self.encode(X)
        z = self.reparameterize()
        X_recon = self.decode(z)
        # ELBO
        KL_per_point = self.KL_divergence()
        loss = self.reconstruction_loss.forward(X, X_recon) + KL_per_point.mean()
        return X_recon, loss
        
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
        
        
        
        
        
