import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from image_tokenizer import ImageTokenizer, InverseImageTokenizer
from text_tokenizer import TextTokenizer, InverseTextTokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_flash_attention=True):
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self.d_model = d_model
        self.num_heads = num_heads
        
        if not use_flash_attention:
            # Regular attention module
            self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        else:
            # For flash attention, we need separate projections
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        
        if not self.use_flash_attention:
            # Regular attention path
            attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
            x = residual + self.dropout(attn_output)
            x = self.norm1(x)
        else:
            # Flash attention path - apply pre-norm
            x_norm = self.norm1(x)
            
            # Project queries, keys, values
            batch_size, seq_len = x_norm.shape[:2]
            q = self.q_proj(x_norm).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x_norm).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x_norm).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
            
            # Apply flash attention
            flash_mask = None
            if mask is not None:
                flash_mask = torch.zeros_like(mask, dtype=torch.bool)
                flash_mask = flash_mask.masked_fill(mask > 0, True)
            
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=flash_mask, 
                dropout_p=self.dropout.p if self.training else 0.0
            )
            
            # Reshape and project back
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            attn_output = self.out_proj(attn_output)
            
            # Residual connection
            x = residual + self.dropout(attn_output)
        
        # Feedforward with residual connection and layer norm
        residual = x
        ff_output = self.ff(self.norm2(x))
        x = residual + ff_output
        
        return x

class MultimodalTransformer(nn.Module):
    def __init__(self, 
                 d_model=512, 
                 num_heads=8, 
                 num_layers=12, 
                 d_ff=2048,
                 dropout=0.1,
                 max_len=5000,
                 image_size=(48, 48),
                 patch_size=2,
                 image_channels=3,
                 bytes_per_token=2,
                 max_text_length=1024,
                 use_flash_attention=True):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Calculate number of patches for an image
        self.patches_h = math.ceil(image_size[0] / patch_size)
        self.patches_w = math.ceil(image_size[1] / patch_size)
        self.num_patches = self.patches_h * self.patches_w
        
        # Tokenizers for different modalities
        self.image_tokenizer = ImageTokenizer(
            in_channels=image_channels,
            out_channels=d_model,
            patch_size=patch_size,
            debug=False
        )
        
        self.inverse_image_tokenizer = InverseImageTokenizer(
            in_channels=d_model,
            out_channels=image_channels,
            patch_size=patch_size,
            debug=False
        )
        
        self.text_tokenizer = TextTokenizer(
            out_channels=d_model,
            bytes_per_token=bytes_per_token,
            max_sequence_length=max_text_length,
            debug=False
        )
        
        self.inverse_text_tokenizer = InverseTextTokenizer(
            in_channels=d_model,
            vocab_size=2**(8*bytes_per_token),
            bytes_per_token=bytes_per_token,
            debug=False
        )
        
        # Input processing
        self.input_proj = nn.Linear(d_model, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer decoder layers
        self.transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout, use_flash_attention)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Causal mask for autoregressive generation
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
        
    def forward(self, x, mode="text_to_text", generate=False, max_new_tokens=None, target_outputs=None):
        """
        x: Input data (text)
        mode: "text_to_text" or "text_to_image"
        generate: Whether to generate output autoregressively
        max_new_tokens: Maximum number of tokens to generate
        target_outputs: Target outputs for teacher forcing during training
        """
        if mode == "text_to_text":
            return self.text_to_text(x, generate, max_new_tokens, target_outputs=target_outputs)
        elif mode == "text_to_image":
            return self.text_to_image(x, generate, max_new_tokens, target_outputs=target_outputs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def generate_from_text(self, text_batch, output_type, generate=False, max_new_tokens=None, target_outputs=None):
        """
        Shared generation method for both text_to_text and text_to_image
        
        text_batch: List of strings or batch of text tokens
        output_type: "text" or "image" to determine the special token
        generate: Whether to generate autoregressively
        max_new_tokens: Maximum number of tokens to generate
        target_outputs: Target outputs for teacher forcing
        """
        batch_size = len(text_batch) if isinstance(text_batch, list) else text_batch.size(0)
        
        

        # Tokenize text - now handling tuple return value
        text_embeddings, text_tokens = self.text_tokenizer(text_batch)
        # Move to the appropriate device
        device = next(self.parameters()).device
        text_embeddings = text_embeddings.to(device)
        text_tokens = text_tokens.to(device)
        text_len = text_embeddings.size(1)
        
        # Create special token for output type (all 0s for text, all 1s for image)
        # Ensure it's on the same device as text_embeddings
        output_type_token = torch.zeros((batch_size, 1, self.d_model), device=device)
        if output_type == "image":
            output_type_token = torch.ones((batch_size, 1, self.d_model), device=device)
        
        # Append the output type token to the text embeddings
        input_embeddings = torch.cat([text_embeddings, output_type_token], dim=1)
        input_len = input_embeddings.size(1)
        
        if not generate:
            # Teacher forcing mode (for training)
            if target_outputs is None:
                raise ValueError("Teacher forcing requires target_outputs to be provided")
            
            # Ensure target_outputs is on the same device
            target_outputs = target_outputs.to(device)
            
            # Prepare full sequence for teacher forcing
            full_sequence = torch.cat([input_embeddings, target_outputs], dim=1)
            
            # Process the sequence
            x = self.input_proj(full_sequence)
            x = self.pos_encoder(x)
            
            # Create causal mask for the entire sequence
            seq_len = x.size(1)
            mask = torch.ones(seq_len, seq_len, device=device) * float('-inf')
            
            # Allow full visibility of input tokens
            mask[:, :input_len] = 0
            
            # For output tokens, apply causal masking
            for j in range(input_len, seq_len):
                mask[j, :j+1] = 0
            
            # Apply transformer blocks
            for block in self.transformer_blocks:
                x = block(x, mask=mask)
            
            # Apply output projection
            outputs = self.output_proj(x)
            
            # We're only interested in the predictions for target positions
            output_embeddings = outputs[:, input_len:, :]
            
            return output_embeddings
        
        else:
            # Autoregressive generation mode
            generated = input_embeddings
            output_tokens = torch.zeros((batch_size, 0, self.d_model), device=device)
            
            # Set max tokens based on output type
            if output_type == "text":
                max_tokens = max_new_tokens if max_new_tokens else self.text_tokenizer.max_sequence_length
            else:  # image
                max_tokens = max_new_tokens if max_new_tokens else self.num_patches
            
            # Generate tokens one by one
            for i in range(max_tokens):
                # Combine input and currently generated output tokens
                combined = torch.cat([generated, output_tokens], dim=1)
                combined = self.input_proj(combined)
                combined = self.pos_encoder(combined)
                
                # Create appropriate mask
                curr_len = combined.size(1)
                mask = torch.ones(curr_len, curr_len, device=device) * float('-inf')
                
                # Allow seeing all input tokens
                mask[:, :input_len] = 0
                
                # Apply causal masking for output tokens
                for j in range(input_len, curr_len):
                    mask[j, :j+1] = 0
                
                # Apply transformer blocks
                for block in self.transformer_blocks:
                    combined = block(combined, mask=mask)
                
                # Get next token prediction (only for the latest position)
                next_token = self.output_proj(combined[:, -1:, :])
                
                # Append to generated output tokens
                output_tokens = torch.cat([output_tokens, next_token], dim=1)
                
                # For text generation, check if we've generated an end token (can be customized)
                if output_type == "text" and i > 0:
                    # Check for special end of text condition
                    # This is placeholder logic - you'd need to implement proper end token detection
                    next_token_logits = self.inverse_text_tokenizer(next_token)
                    next_token_id = torch.argmax(next_token_logits, dim=-1)
                    if (next_token_id == 0).all():  # Assuming 0 is end token
                        break
            
            return output_tokens
    
    def text_to_text(self, text_batch, generate=False, max_new_tokens=None, target_outputs=None):
        """
        text_batch: List of strings or batch of text tokens
        generate: Whether to generate text autoregressively
        max_new_tokens: Maximum number of text tokens to generate
        target_outputs: Target text embeddings for teacher forcing
        """
        output_tokens = self.generate_from_text(
            text_batch, 
            output_type="text",
            generate=generate, 
            max_new_tokens=max_new_tokens,
            target_outputs=target_outputs
        )
        
        if generate:
            # Convert token embeddings to text tokens then to text
            text_logits = self.inverse_text_tokenizer(output_tokens)
            text_ids = torch.argmax(text_logits, dim=-1)
            
            # Convert to text
            generated_text = self.inverse_text_tokenizer.decode(text_ids)
            return generated_text
        else:
            # During training with teacher forcing, return the output embeddings
            return output_tokens
    
    def text_to_image(self, text_batch, generate=False, max_new_tokens=None, target_outputs=None):
        """
        text_batch: List of strings or batch of text tokens
        generate: Whether to generate image autoregressively
        max_new_tokens: Maximum number of image patches to generate
        target_outputs: Target image tokens for teacher forcing
        """
        output_tokens = self.generate_from_text(
            text_batch, 
            output_type="image",
            generate=generate, 
            max_new_tokens=max_new_tokens,
            target_outputs=target_outputs
        )
        
        if generate:
            # Detokenize the generated image tokens
            generated_image = self.inverse_image_tokenizer(
                output_tokens,
                original_size=self.image_size
            )
            return generated_image
        else:
            # During training with teacher forcing, return the output embeddings
            return output_tokens

def create_multimodal_model(
    model_size='base',  # 'small', 'base', 'large', 'xl'
    image_size=(48, 48),
    patch_size=2,
    image_channels=3,
    bytes_per_token=2,
    max_text_length=1024,
    use_flash_attention=True,
    custom_config=None
):
    """
    Create a multimodal model with different size configurations
    """
    if custom_config:
        return MultimodalTransformer(**custom_config)
    
    model_configs = {
        'small': {
            'd_model': 256,
            'num_heads': 4,
            'num_layers': 6,
            'd_ff': 1024,
            'dropout': 0.1
        },
        'base': {
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 12,
            'd_ff': 2048,
            'dropout': 0.1
        },
        'large': {
            'd_model': 768,
            'num_heads': 12,
            'num_layers': 24,
            'd_ff': 3072,
            'dropout': 0.1
        },
        'xl': {
            'd_model': 1024,
            'num_heads': 16,
            'num_layers': 36,
            'd_ff': 4096,
            'dropout': 0.1
        }
    }
    
    config = model_configs[model_size]
    config['image_size'] = image_size
    config['patch_size'] = patch_size
    config['image_channels'] = image_channels
    config['bytes_per_token'] = bytes_per_token
    config['max_text_length'] = max_text_length
    config['use_flash_attention'] = use_flash_attention
    
    model = MultimodalTransformer(**config)
    
    # Print approximate model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Created {model_size} model with approximately {param_count/1e6:.2f}M parameters")
    
    return model
