import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import glob
import numpy as np
# Updated import for autocast to avoid deprecation warning
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
import argparse
import math
import random
from PIL import Image
import torchvision.transforms as transforms

from multimodal_model import create_multimodal_model, MultimodalTransformer
from text_tokenizer import TextTokenizer, InverseTextTokenizer
from image_tokenizer import ImageTokenizer, InverseImageTokenizer
from multimodal_training_utils import save_image_grid
from data_loader import MultiModalDataLoader  # Import the new data loader

# Set up Torch performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Multimodal Transformer model")
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large', 'xl'], 
                        help="Size of the model to train")
    parser.add_argument('--image_size', type=str, default='16,16', 
                        help="Image dimensions (height,width)")
    parser.add_argument('--patch_size', type=int, default=2, 
                        help="Size of image patches")
    parser.add_argument('--image_channels', type=int, default=3, 
                        help="Number of image channels (RGB=3)")
    parser.add_argument('--bytes_per_token', type=int, default=2, 
                        help="Number of bytes per text token")
    parser.add_argument('--max_text_length', type=int, default=1024, 
                        help="Maximum text sequence length")
    parser.add_argument('--batch_size', type=int, default=1, 
                        help="Training batch size")
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=10000, 
                        help="Number of warmup steps for LR scheduler")
    parser.add_argument('--device', type=str, default='cuda', 
                        help="Device to run training on")
    parser.add_argument('--use_flash_attention', type=bool, default=True, 
                        help="Whether to use flash attention mechanism")
    return parser.parse_args()

def main():
    args = parse_args()
    # Check if cuda is available
    if not torch.cuda.is_available():
        args.device = 'cpu'
        args.use_flash_attention = False
    
    # Parse image size
    image_size = tuple(map(int, args.image_size.split(',')))
    
    # Set up training configuration
    model_name = f"MultimodalTransformer_{args.model_size}_{args.lr}"
    device = torch.device(args.device)
    out_dir = f"{model_name}_out"
    checkpoint_dir = f"{model_name}_checkpoints"
    
    # Set up TensorBoard logging
    writer = SummaryWriter(log_dir=f"runs/{model_name}", flush_secs=10)
    
    # Create output directories
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Clean previous output files
    for f in glob.glob(f'{out_dir}/*'):
        os.remove(f)
    
    # Create the multimodal model
    model = create_multimodal_model(
        model_size=args.model_size,
        image_size=image_size,
        patch_size=args.patch_size,
        image_channels=args.image_channels,
        bytes_per_token=args.bytes_per_token,
        max_text_length=args.max_text_length,
        use_flash_attention=args.use_flash_attention
    ).to(device)
    
    # Print model param count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {model_name} has {num_params / 1e6:.2f}M parameters")
    print(f"Model {model_name} is running on {args.device}")

    # Set up optimizer and loss functions
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    # We'll use MSE loss for both text and image reconstruction
    reconstruction_criterion = nn.MSELoss()
    
    # Initialize training state
    step = 0
    total_steps = -1
    start_epoch = 0
    
    # Load checkpoint if available
    checkpoints_sorted = glob.glob(f'{checkpoint_dir}/*.pt')
    if len(checkpoints_sorted) > 0:
        checkpoints_sorted.sort(key=os.path.getmtime)
        print(f"Loading checkpoint {checkpoints_sorted[-1]}")
        checkpoint = torch.load(checkpoints_sorted[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_steps = checkpoint['total_steps']
        step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint {checkpoints_sorted[-1]}, total steps: {total_steps}, step: {step}, epoch: {start_epoch}")
    
    # Setup learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    
    # Initialize tokenizers for evaluation - using the model's tokenizers directly
    text_tokenizer = model.text_tokenizer
    inverse_text_tokenizer = model.inverse_text_tokenizer
    
    # Create dataset for training - use our new MultiModalDataLoader
    dataset = MultiModalDataLoader(batch_size=args.batch_size, image_size=image_size, image_channels=args.image_channels)
    
    # Training loop
    for epoch in range(start_epoch, 5000):
        loop_time = timer()
        
        for i, (captions, images) in enumerate(dataset, step):
            start = timer()
            
            # Move data to device
            images = images.to(device)
            
            # Tokenize captions using the model's tokenizer
            caption_embeddings, caption_tokens = text_tokenizer(captions)
            caption_embeddings = caption_embeddings.to(device)
            caption_tokens = caption_tokens.to(device)
            
            step += 1
            total_steps += 1
            
            # Learning rate warmup
            if total_steps < args.warmup_steps:
                lr = (args.lr / args.warmup_steps) * (total_steps + 1)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Training step
            optimizer.zero_grad()
            
            # Phase 1: Text Generation - Using teacher forcing and end-to-end reconstruction
            text_loss = 0
            if args.device == 'cuda':
                with autocast('cuda'):  # Updated to new format
                    # Convert input text to bytes tensor for reconstruction comparison
                    input_text = captions
                    input_bytes = text_tokenizer.text_to_bytes_tensor(input_text).to(device)
                    
                    # Target is the same text shifted by one position
                    # We'll use the byte representation for direct comparison
                    if len(input_bytes.shape) == 3:  # [batch, seq_len, 1]
                        # Create shifted target (remove first byte, add zero at end)
                        target_bytes = torch.cat([
                            input_bytes[:, 1:, :], 
                            torch.zeros((input_bytes.shape[0], 1, 1), device=device)
                        ], dim=1)
                    
                    # Feed to model with teacher forcing
                    # Use original text embeddings as input context
                    text_outputs = model(input_text, mode="text_to_text", generate=False,
                                         target_outputs=caption_embeddings)
                    
                    # Apply inverse text tokenizer to get reconstructed byte sequences
                    # This now returns the actual reconstructed bytes, not just logits
                    reconstructed_bytes = inverse_text_tokenizer(text_outputs, return_bytes=True)
                    
                    print("input text", input_text)
                    print(f"Reconstructed bytes shape: {reconstructed_bytes.shape}")
                    print(f"Target bytes shape: {target_bytes.shape}")
                    
                    # Make sure shapes match for comparison
                    min_len = min(reconstructed_bytes.shape[1], target_bytes.shape[1])
                    reconstructed_bytes = reconstructed_bytes[:, :min_len, :]
                    target_bytes = target_bytes[:, :min_len, :]
                    
                    # Calculate text reconstruction loss using MSE
                    text_loss = reconstruction_criterion(reconstructed_bytes, target_bytes)
                    
                    # Phase 2: Image Generation - Using teacher forcing
                    # This part remains similar as it's already using reconstruction
                    target_image_embeddings, _ = model.image_tokenizer(images)
                    target_image_embeddings = target_image_embeddings.to(device)
                    
                    # Feed to model with teacher forcing
                    image_outputs = model(captions, mode="text_to_image", generate=False, 
                                          target_outputs=target_image_embeddings)
                    
                    # Generate image from embeddings
                    predicted_image = model.inverse_image_tokenizer(image_outputs, 
                                                                   original_size=model.image_size)
                    
                    # Calculate image loss - ensure both tensors are on the same device
                    image_loss = reconstruction_criterion(predicted_image, images.to(predicted_image.device))
                    
                    # Combined loss
                    loss = text_loss + image_loss
            else:
                # Non-CUDA implementation without autocast
                # Text generation with teacher forcing and reconstruction
                input_text = captions
                input_bytes = text_tokenizer.text_to_bytes_tensor(input_text).to(device)
                
                # Create shifted target
                if len(input_bytes.shape) == 3:  # [batch, seq_len, 1]
                    target_bytes = torch.cat([
                        input_bytes[:, 1:, :], 
                        torch.zeros((input_bytes.shape[0], 1, 1), device=device)
                    ], dim=1)
                
                # Feed through model
                text_outputs = model(input_text, mode="text_to_text", generate=False,
                                    target_outputs=caption_embeddings)
                
                # Apply inverse text tokenizer to get reconstructed byte sequences
                reconstructed_bytes = inverse_text_tokenizer(text_outputs, return_bytes=True)
                
                # Make sure shapes match for comparison
                min_len = min(reconstructed_bytes.shape[1], target_bytes.shape[1])
                reconstructed_bytes = reconstructed_bytes[:, :min_len, :]
                target_bytes = target_bytes[:, :min_len, :]
                
                # Calculate text reconstruction loss
                text_loss = reconstruction_criterion(reconstructed_bytes, target_bytes)
                
                # Image generation with teacher forcing
                target_image_embeddings, _ = model.image_tokenizer(images)
                target_image_embeddings = target_image_embeddings.to(device)
                
                image_outputs = model(captions, mode="text_to_image", generate=False, 
                                      target_outputs=target_image_embeddings)
                
                predicted_image = model.inverse_image_tokenizer(image_outputs, 
                                                               original_size=model.image_size)
                
                # Calculate image loss
                image_loss = reconstruction_criterion(predicted_image, images.to(predicted_image.device))
                
                # Combined loss
                loss = text_loss + image_loss
            
            loss.backward()
            loss_item = loss.item()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Generate samples periodically
            if total_steps % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    # Remove image-to-text generation section
                    
                    # Generate image from a random caption
                    # No device issues here as captions is a list of strings
                    generated_image = model(captions[0:1], mode="text_to_image", generate=True)
                    
                    # Save generated image
                    img_transform = transforms.ToPILImage()
                    img = img_transform(generated_image[0].cpu())
                    img.save(f"{out_dir}/e_{str(epoch)}_i_{str(total_steps)}_generated.png")
                    
                    # Save target image for comparison
                    target_img = img_transform(images[0].cpu())
                    target_img.save(f"{out_dir}/e_{str(epoch)}_i_{str(total_steps)}_target.png")
                model.train()
            
            # Log metrics periodically
            if total_steps % 50 == 0:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                    break
                writer.add_scalar("loss", loss_item, total_steps)
                writer.add_scalar("text_loss", text_loss.item(), total_steps)
                writer.add_scalar("image_loss", image_loss.item(), total_steps)
                writer.add_scalar("lr", lr, total_steps)
                
                print(f"epoch {epoch}, step: {step}, total_steps: {total_steps} "
                      f"loss: {loss_item:.4f}, text_loss: {text_loss.item():.4f}, "
                      f"image_loss: {image_loss.item():.4f}, lr: {lr:.6f}")
            
            # Save checkpoint periodically
            if step % 1000 == 0 and step != 0:
                # Keep only the most recent checkpoint
                old_checkpoints = glob.glob(f"{checkpoint_dir}/*")
                old_checkpoints.sort(key=os.path.getmtime)
                for f in old_checkpoints[:-1]:
                    os.remove(f)
                    
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_steps': total_steps,
                    'loss': loss
                }, f"{checkpoint_dir}/{str(epoch)}_{str(i)}.pt")
        
        # Reset step counter at the end of each epoch
        step = 0

if __name__ == "__main__":
    main()
