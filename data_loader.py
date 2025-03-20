import torch
import random

class MultiModalDataLoader:
    def __init__(self, batch_size=1, num_samples=100, image_size=(48, 48), image_channels=3):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.image_size = image_size
        self.image_channels = image_channels
        self.sample_captions = [
            "A beautiful sunset over the mountains",
            "A cute cat playing with a ball of yarn",
            "A serene lake surrounded by pine trees",
            "An astronaut floating in space with Earth in the background",
            "A delicious plate of pasta with tomato sauce"
        ]
        self.current_sample = 0
        
    def __iter__(self):
        self.current_sample = 0
        return self
    
    def __next__(self):
        if self.current_sample >= self.num_samples:
            raise StopIteration
        
        # Generate batch of dummy data
        batch_captions = []
        batch_images = []
        
        for _ in range(self.batch_size):
            # Use a fixed test caption
            caption = "this is a test caption"
            batch_captions.append(caption)
            
            # Create a random RGB image instead of zeros
            image = torch.rand(self.image_channels, self.image_size[0], self.image_size[1])
            batch_images.append(image)
            
        self.current_sample += 1
        return batch_captions, torch.stack(batch_images)
        
    def __len__(self):
        return self.num_samples
