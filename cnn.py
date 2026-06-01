import numpy as np

class CNN:
    def __init__(self, batch_size, rows, cols, in_channels, out_channels, kernel_size, padding) -> None:
        # outchannels = (K, 1)
        self.bs = batch_size 
        self.width = cols
        self.height = rows
        self.depth = in_channels
        self.kernel_size = kernel_size
        self.kernels = []
        self.bias = []
        for _ in range(out_channels):
            self.kernels.append(np.random.randn(kernel_size, kernel_size))
            self.bias.append(np.random.randn())
        self.padding = padding
        
    
    def forward(self, X):
        # X has shape (N, C, H, W)
        # add padding to image:
        for image in range(self.bs):
            for channel in range(self.depth):
                for r in range(self.height - self.kernel_size):
                    for c in range(self.width - self.kernel_size):
                        
        
            
                
                
        return
            
            
        
        
    