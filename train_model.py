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
    text_criterion = nn.CrossEntropyLoss()
    image_criterion = nn.MSELoss()
    
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
            
            # Phase 1: Text Generation - Get first token and predict the rest
            # We'll use teacher forcing during training
            text_loss = 0
            if args.device == 'cuda':
                with autocast('cuda'):  # Updated to new format
                    # For text generation training, we use all but the last token as input
                    # and predict the next token in sequence
                    text_input = caption_embeddings[:, :-1, :]
                    text_target = caption_tokens[:, 1:]  # Shift by 1 to get next token
                    
                    # Get token predictions from model - using the inverse_text_tokenizer directly
                    text_logits = inverse_text_tokenizer(text_input)
                    
                    # Calculate text loss
                    text_loss = text_criterion(text_logits.view(-1, text_logits.size(-1)), text_target.view(-1))
                    
                    # Phase 2: Image Generation - Using generation mode for training to avoid NotImplementedError
                    # We'll use the generated image and compare it with the target
                    predicted_image = model(captions, mode="text_to_image", generate=True)
                    
                    # Calculate image loss
                    image_loss = image_criterion(predicted_image, images)
                    
                    # Combined loss
                    loss = text_loss + image_loss
            else:
                # Non-CUDA implementation without autocast
                # Text generation training
                text_input = caption_embeddings[:, :-1, :]
                text_target = caption_tokens[:, 1:]
                text_logits = inverse_text_tokenizer(text_input)
                text_loss = text_criterion(text_logits.view(-1, text_logits.size(-1)), text_target.view(-1))
                
                # Image generation training - same approach as with CUDA
                predicted_image = model(captions, mode="text_to_image", generate=True)
                image_loss = image_criterion(predicted_image, images)
                
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
                    # Generate text from a random image in the batch
                    generated_text = model(images[0:1], mode="image_to_text", generate=True, max_new_tokens=50)
                    print(f"Generated text: {generated_text}")
                    
                    # Generate image from a random caption
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
