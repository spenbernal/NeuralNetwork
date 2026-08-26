import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tensorflow.keras.datasets import mnist 
import numpy as np
import matplotlib.pyplot as plt
import deeplearning.vae as v
import deeplearning.layers as l
import csv
from pathlib import Path

test_dir = Path(__file__).parent
print('Parent Directory', test_dir)
result_dir = test_dir / 'vae_results'
print('Result Directory', result_dir)
# Logger
class Logger:
    def __init__(self, results_dir) -> None:
        self.results_dir = results_dir
        
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.filepath = self.results_dir / "vae_training_metrics.csv"
        
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "loss",
                "reconstruction_loss",
                "kl_loss"
            ])
            
    def log(self, epoch, loss, recon_loss, kl_loss):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                epoch,
                loss,
                recon_loss,
                kl_loss
            ])

# Reconstruction Loss
class BCE:
    def forward(self, y, logit):
            # logit: scalar
            self.y = y
            self.prob = 1 / (1 + np.exp(-logit))
            self.prob = np.clip(self.prob, 1e-8, 1 - 1e-8)
            
            loss_per_point = -(y * np.log(self.prob) + (1 - y)* np.log(1 - self.prob)).sum(axis= (1,2,3))
            return loss_per_point.mean()
            
    def backwards(self):
        return (self.prob - self.y) / self.y.shape[0]
    
# Class for the MLP of the encoder
class MLP:
    def __init__(self, linear_layers: list[l.Layer], heads) -> None:
        self.layers = linear_layers
        self.heads = heads
        
    def forward(self, X):
            for layer in self.layers:
                X = layer.forward(X)
                
            outputs = [head.forward(X) for head in self.heads]
            # mu, logvar
            return outputs
        
    def backwards(self, delta_mu, delta_logvar):
        
        delta = (self.heads[0].backwards(delta_mu) 
                        + self.heads[1].backwards(delta_logvar))
        for layer in reversed(self.layers):
            delta = layer.backwards(delta) 
        return delta
        
    def update(self, eta):
        for layer in self.layers:
            layer.update(eta)
            
        for head in self.heads:
            head.update(eta)
            
        return
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
            
        for head in self.heads:
            head.zero_grad()

# Class for the encoder
class CNN:
    def __init__(self, conv_layers: list[l.Layer]) -> None:
            self.conv_layers = conv_layers
            
    def forward(self, X: np.ndarray):
        # N, C_in, H, W
        Z = X
        for layer in self.conv_layers:
            Z = layer.forward(Z)
        return Z
    
    def backwards(self, delta):
        for layer in reversed(self.conv_layers):
            delta = layer.backwards(delta)
        return delta
    
    def update(self, eta):
        for layer in self.conv_layers:
            layer.update(eta)
        
    def zero_grad(self):
        for layer in self.conv_layers:
            layer.zero_grad()
# Class for the encoder (X -> z)
class Encoder:
    def __init__(self, cnn: CNN, mlp: MLP) -> None:
        self.cnn = cnn
        self.mlp = mlp
    
    def forward(self, X):
        Z = self.cnn.forward(X)
        mu, logvar = self.mlp.forward(Z)
    
        return mu, logvar
    
    def backwards(self, delta_mu, delta_logvar):
        delta_mlp = self.mlp.backwards(delta_mu, delta_logvar)
        delta_X = self.cnn.backwards(delta_mlp)
        return delta_X
    
    def zero_grad(self):
        self.cnn.zero_grad()
        self.mlp.zero_grad()
    
    def update(self, eta):
        self.mlp.update(eta)
        self.cnn.update(eta)

# Class for the decoder (z -> X_hat)
class Decoder:
    def __init__(self, linear_layers, conv_layers, reshape_shape) -> None:
        self.linear_layers = linear_layers
        self.conv_layers = conv_layers
        self.reshape_shape = reshape_shape
    
    def forward(self, Z):
        # Z: N,L
        self.N = Z.shape[0]
        X = Z
        
        for layer in self.linear_layers:
            X = layer.forward(X)
            
        X = X.reshape(self.N, *self.reshape_shape)
        
        for layer in self.conv_layers:
            X = layer.forward(X)
            
        return X

    def backwards(self, delta):
        for layer in reversed(self.conv_layers):
            delta = layer.backwards(delta)
        
        delta = delta.reshape(self.N, -1)
        for layer in reversed(self.linear_layers):
            delta = layer.backwards(delta)
            
        return delta

    def zero_grad(self):
        for layer in self.linear_layers:
            layer.zero_grad()
        for layer in self.conv_layers:
            layer.zero_grad()
    
    def update(self, eta):
        for layer in self.linear_layers:
            layer.update(eta)
            
        for layer in self.conv_layers:
            layer.update(eta)
                

def normalize(X):
    # normalize data
    return (X - X.min())/(X.max() - X.min())

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Normalization
    X_train, X_test = normalize(X_train), normalize(X_test)
    # Add channel dimensions
    X_train, X_test = X_train[:, None, :, :], X_test[:, None, :, :]
    
    return (X_train, y_train), (X_test, y_test)

def init_cnn():
    cnn_layers = [l.ConvolutionLayer(in_channels= 1, out_channels= 8, kernel_size= 2, padding= 1),
                  l.MaxPooling(k= 2),
                  l.ConvolutionLayer(in_channels= 8, out_channels= 32, kernel_size= 2, padding= 1),
                  l.MaxPooling(k= 2),
                  l.Flatten()]
    cnn = CNN(cnn_layers)
    return cnn

def init_mlp(init_dim, latent_dim):
    linear_layers = [l.Linear(init_dim, 64),
                     l.ReLU()]
    heads = [l.Linear(64, latent_dim), l.Linear(64, latent_dim)]
    mlp = MLP(linear_layers= linear_layers, heads= heads)
    return mlp

# Training loop
def train(X, vae, batch_size, epochs, eta, logger: Logger):
    N = X.shape[0]
    for e in range(epochs):
        # shuffle data then split into batches
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        X_batches = [X[i:i+batch_size] for i in range(0, N, batch_size)]
        loss = 0
        loss_recon = 0
        loss_kl = 0
        
        print(f'---------- Epoch {e+1} ----------')
        # loop over each batch 
        for idx, batch in enumerate(X_batches):
            X_recon, batch_loss, recon, kl = vae.forward(batch) 
            # Log statistics per batch
            print(f'Batch {idx + 1} | Total Loss: {batch_loss:.5f} \
                  | Reconstruction Loss: {recon:.5f} | KL Divergence: {kl:.5f}')
            # unaverage per batch loss metrics so we can average over all batches for a full epoch
            bs = batch.shape[0]
            
            loss += batch_loss * bs
            loss_recon += recon * bs
            loss_kl += kl * bs
            
            # zero out gradients, compute backwards pass, update parameters
            
            vae.zero_grad()
            print(
                "mu:",
                np.min(vae.mu),
                np.max(vae.mu),
                np.mean(vae.mu)
            )

            print(
                "logvar:",
                np.min(vae.logvar),
                np.max(vae.logvar),
                np.mean(vae.logvar)
            )
            delta_X = vae.backwards()
            vae.update(eta)
        # compute avg loss metrics for epoch
        loss /= N
        loss_recon /= N
        loss_kl /= N
        # log epoch loss statistics
        logger.log(e+1, loss, loss_recon, loss_kl)
        print('***************')
        print(f'Total Loss: {loss:.5f} \
              | Reconstruction Loss {loss_recon:.5f} \
              | KL Divergence: {loss_kl:.5}')
        print('***************')
    
    # return the learned vae
    return vae
        
    
    
def initialize_and_train():
    logger = Logger(result_dir)
    
    # train, test sets preprocessed
    latent_dim = 16
    (X_train, y_train), (X_test, y_test) = load_data()
    
    # reconstruction loss
    recon_loss = BCE()
    
    # initialize cnn and run dummy point to get dimension of data that will be passed to MLP
    cnn = init_cnn()
    dummy = np.zeros_like(X_train[:1])
    Z = cnn.forward(dummy)
    flatten_dim = Z.shape[-1] # type: ignore
    print('Flattened Dim: ', flatten_dim)
    # init mlp
    mlp = init_mlp(flatten_dim, latent_dim)
    # init encoder
    encoder = Encoder(cnn, mlp)
    
    # init decoder layers
    decoder_linear_layers = [l.Linear(latent_dim, 32*7*7), l.ReLU()]
    decoder_conv_layers = [l.Upsample(scale= 2), 
                           l.ConvolutionLayer(in_channels= 32, out_channels= 8, kernel_size= 3, padding= 1),
                           l.ReLU(),
                           l.Upsample(scale= 2),
                           l.ConvolutionLayer(in_channels= 8, out_channels= 1, kernel_size= 3, padding= 1)]
    decoder_reshape = (32, 7, 7)
    # init decoder
    decoder = Decoder(decoder_linear_layers, decoder_conv_layers, decoder_reshape)
    
    # init VAE
    vae = v.VAE(encoder, decoder, latent_dim, recon_loss)
    
    # init training loop parameters
    eta = 0.0001
    batch_size = 64
    epochs = 6
    
    # train VAE
    train(X_train, vae, batch_size, epochs, eta, logger)
    
    
    return vae
    
def generate_images(vae: v.VAE):
    # Plot reconstructed images sampled from the latent space
    # generate 5 examples
    images = vae.generate(5) # 5, 1, 28, 28
    fig, axes = plt.subplots(1, 5)
    fig.suptitle('Generated Images from Latent Space')
    
    for i in range(5):
        img = images[i, :, :, :].squeeze()
        axes[i].imshow(img, cmap= 'gray')
        axes[i].axis('off')
        axes[i].set_title(f'Latent Image {i}') 
    
    plot_file = result_dir / "vae_generated_samples.png"
        
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    '''
    initialize all models, train vae, plot loss metrics, generate images from vae 
    '''
    
    vae = initialize_and_train()
    generate_images(vae)
    
if __name__ == '__main__':
    main()
    