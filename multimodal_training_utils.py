import torch
import torch.nn as nn
import random
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import io
import os

def save_image_grid(images, filepath, nrow=4, title=None):
    """
    Save a grid of images for visualization
    
    Args:
        images: tensor of shape [B, C, H, W]
        filepath: path to save the image grid
        nrow: number of images per row
        title: optional title for the grid
    """
    # Convert to PIL images
    pil_images = []
    for img in images:
        # Normalize to [0, 1] range if needed
        if img.min() < 0 or img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min())
        
        # Convert to PIL
        transform = transforms.ToPILImage()
        pil_img = transform(img.cpu())
        pil_images.append(pil_img)
    
    # Determine grid size
    num_images = len(pil_images)
    ncol = (num_images + nrow - 1) // nrow  # Ceiling division
    
    # Create a blank canvas
    width, height = pil_images[0].size
    grid = Image.new('RGB', (width * ncol, height * nrow))
    
    # Paste images into grid
    for i, img in enumerate(pil_images):
        row = i // ncol
        col = i % ncol
        grid.paste(img, (col * width, row * height))
    
    # Add title if provided
    if title:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(grid)
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
        # Save with matplotlib
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    else:
        # Save directly
        grid.save(filepath)
    
    return grid

def get_next_token_prediction(model, sequence, tokenizer, inverse_tokenizer, device):
    """
    Get the model's prediction for the next token given a sequence
    
    Args:
        model: the multimodal transformer model
        sequence: tensor of token embeddings [1, seq_len, d_model]
        tokenizer: text tokenizer instance
        inverse_tokenizer: inverse text tokenizer instance
        device: device to run prediction on
        
    Returns:
        next_token: the predicted next token
        next_token_embed: embedding of the predicted token
    """
    with torch.no_grad():
        # Get model prediction
        logits = inverse_tokenizer(sequence)
        
        # Get most likely token
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        
        # Convert token to embedding
        next_token_embed, _ = tokenizer([next_token.item()])
        next_token_embed = next_token_embed.to(device)
        
    return next_token.item(), next_token_embed

def generate_text_from_prompt(model, prompt, tokenizer, inverse_tokenizer, max_length=50, device='cuda'):
    """
    Generate text from a given prompt
    
    Args:
        model: the multimodal transformer model
        prompt: the starting text prompt
        tokenizer: text tokenizer instance
        inverse_tokenizer: inverse text tokenizer instance
        max_length: maximum number of tokens to generate
        device: device to run generation on
        
    Returns:
        generated_text: the complete generated text
    """
    model.eval()
    
    # Tokenize the prompt
    prompt_embed, prompt_tokens = tokenizer([prompt])
    prompt_embed = prompt_embed.to(device)
    
    # Initialize sequence with prompt
    sequence = prompt_embed
    all_tokens = prompt_tokens.tolist()[0]
    
    # Generate tokens one by one
    for _ in range(max_length):
        next_token, next_token_embed = get_next_token_prediction(
            model, sequence, tokenizer, inverse_tokenizer, device
        )
        
        # Add token to sequence
        all_tokens.append(next_token)
        sequence = torch.cat([sequence, next_token_embed.unsqueeze(0)], dim=1)
        
        # Stop if end token is generated
        if next_token == 0:
            break
    
    # Convert tokens back to text
    generated_text = tokenizer.binary_tokens_to_text([all_tokens])
    
    model.train()
    return generated_text[0]

def autoregressive_text_to_image(model, caption, image_size=(48, 48), device='cuda'):
    """
    Generate an image from a caption autoregressively
    
    Args:
        model: the multimodal transformer model
        caption: the text caption
        image_size: size of the image to generate
        device: device to run generation on
        
    Returns:
        generated_image: the generated image tensor
    """
    model.eval()
    
    with torch.no_grad():
        # Generate image
        generated_image = model(
            [caption], 
            mode="text_to_image", 
            generate=True, 
            max_new_tokens=model.num_patches
        )
    
    model.train()
    return generated_image
