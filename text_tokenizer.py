import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextTokenizer(nn.Module):
    def __init__(
        self,
        out_channels=512,
        bytes_per_token=2,  # Changed from kernel_size to bytes_per_token
        max_sequence_length=1024,
        debug=False
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # First layer: process bytes_per_token bytes at a time
            nn.Conv1d(
                in_channels=1,
                out_channels=out_channels//2,
                kernel_size=bytes_per_token,
                stride=bytes_per_token
            ),
            nn.GELU(),
            nn.BatchNorm1d(out_channels//2),
            
            # Second layer: combine pairs of tokens
            nn.Conv1d(
                in_channels=out_channels//2,
                out_channels=out_channels//2,
                kernel_size=2,
                stride=2
            ),
            nn.GELU(),
            nn.BatchNorm1d(out_channels//2),
            
            # Third layer: combine pairs again
            nn.Conv1d(
                in_channels=out_channels//2,
                out_channels=out_channels,
                kernel_size=2,
                stride=2
            ),
            nn.GELU(),
            nn.BatchNorm1d(out_channels)
        )
        
        self.bytes_per_token = bytes_per_token  # Store for future use
        self.out_channels = out_channels
        self.max_sequence_length = max_sequence_length
        self.debug = debug
    
    def text_to_bytes_tensor(self, text_batch):
        """
        Convert a batch of text strings to byte tensors
        text_batch: list of strings
        returns: tensor of shape [batch_size, sequence_length, 1]
        """
        batch_bytes = []
        
        for text in text_batch:
            # Convert text to bytes
            byte_data = text.encode('utf-8')
            
            # Pad to max_sequence_length if needed
            if len(byte_data) > self.max_sequence_length:
                byte_data = byte_data[:self.max_sequence_length]
            else:
                byte_data = byte_data + b'\x00' * (self.max_sequence_length - len(byte_data))
            
            # Convert bytes to integers
            bytes_as_ints = [b for b in byte_data]
            batch_bytes.append(bytes_as_ints)
        
        # Create tensor with shape [batch_size, sequence_length, 1]
        return torch.tensor(batch_bytes, dtype=torch.float32).unsqueeze(-1)
    
    def bytes_tensor_to_text(self, bytes_tensor):
        """
        Convert a batch of byte tensors back to text
        bytes_tensor: tensor of shape [batch_size, sequence_length, 1]
        returns: list of strings
        """
        texts = []
        
        for tensor in bytes_tensor:
            # Convert tensor to bytes
            byte_values = [int(b.item()) for b in tensor.squeeze(-1)]
            byte_data = bytes(byte_values)
            
            # Decode bytes to text, ignoring errors and stripping null bytes
            text = byte_data.decode('utf-8', errors='ignore').rstrip('\x00')
            texts.append(text)
        
        return texts
    
    def forward(self, text_batch):
        """
        Convert text to tokenized representation
        text_batch: list of strings or tensor of shape [batch_size, sequence_length, 1]
        returns: tuple of (embeddings, tokens) where:
            - embeddings: tensor of shape [batch_size, reduced_length, out_channels]
            - tokens: tensor of shape [batch_size, sequence_length]
        """
        if isinstance(text_batch, list) and isinstance(text_batch[0], str):
            # Convert text to bytes tensor
            x = self.text_to_bytes_tensor(text_batch)
        else:
            # Already a tensor
            x = text_batch
            
        if self.debug:
            print(f"Input tensor shape: {x.shape}")
        
        # Store the original tokens for return
        tokens = x.squeeze(-1).long()
        
        # Permute to [batch, channels, length] for Conv1d
        x_permuted = x.permute(0, 2, 1)
        
        if self.debug:
            print(f"Permuted tensor shape: {x_permuted.shape}")
        
        # Get the device of the model parameters and move input tensor to that device
        device = next(self.parameters()).device
        x_permuted = x_permuted.to(device)
        
        # Apply encoding
        encoded = self.encoder(x_permuted)
        
        # Permute back to [batch, length, channels]
        encoded = encoded.permute(0, 2, 1)
        
        # Ensure output tensor is contiguous
        encoded = encoded.contiguous()
        
        if self.debug:
            print(f"Encoded tensor shape: {encoded.shape}")
        
        return encoded, tokens


class InverseTextTokenizer(nn.Module):
    def __init__(
        self,
        in_channels=512,
        out_channels=1,
        bytes_per_token=2,  # Changed from kernel_size to bytes_per_token
        vocab_size=None,    # Added vocab_size parameter
        debug=False
    ):
        super().__init__()
        
        # If vocab_size is not provided, calculate it from bytes_per_token
        if vocab_size is None:
            vocab_size = 2**(8*bytes_per_token)
            
        self.vocab_size = vocab_size
        
        self.decoder = nn.Sequential(
            # First layer: expand channels
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels//2,
                kernel_size=1,
                stride=1
            ),
            nn.GELU(),
            nn.BatchNorm1d(in_channels//2),
            
            # Second layer: upsample by 2
            nn.Conv1d(
                in_channels=in_channels//2,
                out_channels=in_channels//4 * 2,  # * 2 for upsampling
                kernel_size=1,
                stride=1
            ),
            nn.GELU(),
            nn.BatchNorm1d(in_channels//4 * 2),
            
            # Third layer: upsample by 2 again and project to vocab size
            nn.Conv1d(
                in_channels=in_channels//4 * 2,
                out_channels=vocab_size,  # Output dimension is vocab_size for classification
                kernel_size=1,
                stride=1
            )
        )
        
        self.bytes_per_token = bytes_per_token  # Store for consistency
        self.expansion_factor = 8  # Total expansion: 2*2*2 = 8
        self.debug = debug
    
    def forward(self, x, tokenizer=None, return_bytes=False):
        """
        x: [batch_size, sequence_length, in_channels]
        tokenizer: Optional tokenizer for converting to text
        return_bytes: If True, return raw byte tensor for reconstruction loss
        returns: logits of shape [batch_size, sequence_length, vocab_size] or
                 byte tensor of shape [batch_size, sequence_length, 1]
        """
        # Permute to [batch, channels, length] for Conv1d
        x_permuted = x.permute(0, 2, 1)
        
        if self.debug:
            print(f"Input tensor shape: {x_permuted.shape}")
        
        # Apply decoding
        decoded = self.decoder(x_permuted)
        
        if self.debug:
            print(f"Decoded tensor shape: {decoded.shape}")
        
        # Permute back to [batch, length, vocab_size]
        logits = decoded.permute(0, 2, 1)
        
        # Ensure output tensor is contiguous
        logits = logits.contiguous()
        
        if self.debug:
            print(f"Output logits shape: {logits.shape}")
        
        # If return_bytes is True, convert logits to byte values
        if return_bytes:
            # Get the most likely byte for each position
            byte_values = torch.argmax(logits, dim=-1).float()
            # Reshape to [batch, seq_len, 1] to match input format
            byte_tensor = byte_values.unsqueeze(-1)
            return byte_tensor
        
        # If tokenizer provided, convert to text
        if tokenizer is not None:
            text_ids = torch.argmax(logits, dim=-1)
            text = tokenizer.bytes_tensor_to_text(text_ids.unsqueeze(-1))
            return text
        
        return logits
    
    def decode(self, token_ids):
        """
        Convert token IDs back to text
        token_ids: tensor of shape [batch_size, sequence_length]
        returns: list of strings
        """
        # This is a placeholder - implement actual decoding logic
        # For now, just return a dummy string
        batch_size = token_ids.shape[0]
        return ["decoded text"] * batch_size


if __name__ == "__main__":
    # Test the tokenizer with sample text
    sample_texts = [
        "Hello, world! This is a test of the text tokenizer.",
        "Using convolutional layers for text is unconventional but interesting."
    ]
    
    # Initialize tokenizer and inverse tokenizer with debug=True
    tokenizer = TextTokenizer(out_channels=512, bytes_per_token=2, debug=True)
    inverse_tokenizer = InverseTextTokenizer(in_channels=512, out_channels=1, bytes_per_token=2, debug=True)
    
    # Tokenize and reconstruct
    encoded, tokens = tokenizer(sample_texts)
    print(f"Encoded shape: {encoded.shape}")
    
    reconstructed_logits = inverse_tokenizer(encoded)
    print(f"Reconstructed logits shape: {reconstructed_logits.shape}")
    
    # Convert reconstructed logits back to text
    reconstructed_texts = inverse_tokenizer(reconstructed_logits, tokenizer=tokenizer)
    
    # Print results
    for i, (original, reconstructed) in enumerate(zip(sample_texts, reconstructed_texts)):
        print(f"\nSample {i+1}:")
        print(f"Original: {original}")
        print(f"Reconstructed: {reconstructed}")
        
    # Calculate reconstruction error
    original_tensor = tokenizer.text_to_bytes_tensor(sample_texts)
    error = torch.mean((original_tensor[:, :reconstructed_logits.shape[1], :] - reconstructed_logits) ** 2).item()
    print(f"\nMean squared reconstruction error: {error:.6f}")
