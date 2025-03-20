import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTokenizer(nn.Module):
    def __init__(
        self,
        in_channels=3,  # RGB by default
        out_channels=512,  # Embedding dimension
        patch_size=2,  # Default 2x2 patches
        debug=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.debug = debug
        
        # Replace linear projection with sequential convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels*2,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.GELU(),
            nn.BatchNorm2d(out_channels*2),
            nn.Conv2d(
                in_channels=out_channels*2,
                out_channels=out_channels,
                kernel_size=1,
                stride=1
            )
        )
        
    def forward(self, x):
        """
        x: [batch_size, channels, height, width]
        returns: tuple of ([batch_size, num_patches, out_channels], image_tokens)
        where image_tokens is a flattened representation of the input
        """
        batch_size, channels, height, width = x.shape
        
        if self.debug:
            print(f"Input shape: {x.shape}")
        
        # Make sure dimensions are divisible by patch_size
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            pad_h = self.patch_size - (height % self.patch_size) if height % self.patch_size != 0 else 0
            pad_w = self.patch_size - (width % self.patch_size) if width % self.patch_size != 0 else 0
            x = F.pad(x, (0, pad_w, 0, pad_h))
            if self.debug:
                print(f"Padded shape: {x.shape}")
        
        # Store input tokens for return (flattened image representation)
        # This is used for consistency with TextTokenizer
        image_tokens = x.reshape(batch_size, -1)
        
        # Apply convolutional encoding
        encoded = self.encoder(x)
        
        if self.debug:
            print(f"After convolution: {encoded.shape}")
        
        # Reshape from [batch, channels, height, width] to [batch, height*width, channels]
        encoded_h, encoded_w = encoded.shape[2], encoded.shape[3]
        embeddings = encoded.permute(0, 2, 3, 1).contiguous().view(batch_size, encoded_h * encoded_w, -1)
        
        if self.debug:
            print(f"Output embeddings: {embeddings.shape}")
        
        return embeddings, image_tokens

class InverseImageTokenizer(nn.Module):
    def __init__(
        self,
        in_channels=512,  # Embedding dimension
        out_channels=3,  # RGB by default
        patch_size=2,  # Default 2x2 patches
        debug=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.debug = debug
        
        # Replace linear projection with sequential layers
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels*patch_size*patch_size*2,
                kernel_size=1,
                stride=1
            ),
            nn.GELU(),
            nn.BatchNorm2d(out_channels*patch_size*patch_size*2),
            nn.Conv2d(
                in_channels=out_channels*patch_size*patch_size*2,
                out_channels=out_channels*patch_size*patch_size,
                kernel_size=1,
                stride=1
            )
        )
    
    def forward(self, x, original_size=None):
        """
        x: [batch_size, num_patches, in_channels]
        original_size: (height, width) of the original image
        returns: [batch_size, out_channels, height, width]
        """
        batch_size, num_patches, channels = x.shape
        
        if self.debug:
            print(f"Input embeddings: {x.shape}")
        
        # Determine grid dimensions based on number of patches
        patches_per_side = int(num_patches ** 0.5)  # Assuming square images
        
        # Reshape to [batch, channels, height, width] for Conv2d
        x_reshaped = x.view(batch_size, patches_per_side, patches_per_side, channels)
        x_reshaped = x_reshaped.permute(0, 3, 1, 2).contiguous()
        
        if self.debug:
            print(f"Reshaped for conv: {x_reshaped.shape}")
        
        # Apply decoding
        decoded = self.decoder(x_reshaped)
        
        if self.debug:
            print(f"After decoder: {decoded.shape}")
        
        # Reshape the output to reconstruct the image
        # [batch, out_channels*patch_size*patch_size, h, w] -> [batch, out_channels, h*patch_size, w*patch_size]
        batch_size, channels, h, w = decoded.shape
        output = decoded.view(
            batch_size, 
            self.out_channels, 
            self.patch_size, 
            h, 
            self.patch_size, 
            w
        ).permute(0, 1, 3, 2, 5, 4).contiguous()
        
        output = output.view(
            batch_size, 
            self.out_channels, 
            h * self.patch_size, 
            w * self.patch_size
        )
        
        # Resize to original dimensions if provided
        if original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        if self.debug:
            print(f"Output shape: {output.shape}")
        
        return output
