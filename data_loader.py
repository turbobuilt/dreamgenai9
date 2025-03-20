import torch
import random
import pandas as pd
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import os
import multiprocessing
from multiprocessing import Pool
from queue import Queue, Empty
import threading
import time

class MultiModalDataLoader:
    def __init__(self, batch_size=1, num_samples=100, image_size=(48, 48), image_channels=3, pool_size=None, prefetch_factor=3):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.image_size = image_size
        self.image_channels = image_channels
        self.pool_size = pool_size if pool_size is not None else max(1, multiprocessing.cpu_count() - 1)
        self.prefetch_factor = prefetch_factor
        
        # COCO dataset URL
        self.coco_url = 'https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/resolve/main/mscoco.parquet'
        self.parquet_file = 'mscoco.parquet'
        
        # Fallback sample captions
        self.sample_captions = [
            "A beautiful sunset over the mountains",
            "A cute cat playing with a ball of yarn",
            "A serene lake surrounded by pine trees",
            "An astronaut floating in space with Earth in the background",
            "A delicious plate of pasta with tomato sauce"
        ]
        
        # Download and prepare dataset
        self.prepare_dataset()
        
        # For cleanup
        self.pool = None
        self.prefetch_queue = None
        self.prefetch_thread = None
        self.stop_event = None
        
    def prepare_dataset(self):
        # Download the parquet file if not present
        if not os.path.exists(self.parquet_file):
            print(f"Downloading COCO dataset metadata from {self.coco_url}")
            r = requests.get(self.coco_url, stream=True)
            with open(self.parquet_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        # Load metadata and clean
        df = pd.read_parquet(self.parquet_file)
        df['normURL'] = df['URL'].str.strip().str.lower()
        df = df.drop_duplicates(subset=['normURL'])
        
        # Create tasks list with URL and caption
        self.tasks = [(i, row['URL'], row['TEXT']) for i, row in df.iterrows()]
        self.num_tasks = len(self.tasks)
        print(f"Loaded {self.num_tasks} image-caption pairs")
        
    @staticmethod
    def process_image(task):
        idx, url, caption = task
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                return None, None
                
            img = Image.open(BytesIO(r.content))
            
            # Process the image
            if img.width < 48 or img.height < 48:  # Assuming minimal dimensions needed
                return None, None
                
            # Crop center
            width, height = img.size
            new_edge = min(width, height)
            left = (width - new_edge) // 2
            top = (height - new_edge) // 2
            img = img.crop((left, top, left + new_edge, top + new_edge))
            
            # Resize to desired dimensions
            img = img.resize((48, 48), Image.LANCZOS)
            img = img.convert("RGB")
            
            # Convert to torch tensor and normalize to [0,1]
            img_arr = np.array(img).astype(np.float32) / 255.0
            # Change from HWC to CHW format for PyTorch
            img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1))
            
            return img_tensor, caption
        except Exception as e:
            return None, None
        
    def prefetch_worker(self):
        """Thread that continuously fetches processed images and puts them in the queue"""
        batch_size = self.batch_size * self.prefetch_factor
        
        while not self.stop_event.is_set():
            # Submit batch of tasks to the pool
            indices = [random.randint(0, self.num_tasks-1) for _ in range(batch_size)]
            tasks = [self.tasks[i] for i in indices]
            
            # Process batch of tasks and add valid results to queue
            results = self.pool.map(self.process_image, tasks)
            for result in results:
                if result[0] is not None:
                    try:
                        self.prefetch_queue.put(result, block=True, timeout=1.0)
                    except:
                        if self.stop_event.is_set():
                            return
            
            # Don't hog CPU if queue is full
            if self.prefetch_queue.qsize() > batch_size * 2:
                time.sleep(0.1)
    
    def __iter__(self):
        # Clean up any existing resources
        self.cleanup()
        
        # Create new resources for this iteration
        self.stop_event = threading.Event()
        self.prefetch_queue = Queue(maxsize=self.batch_size * self.prefetch_factor * 2)
        self.pool = Pool(processes=self.pool_size)
        
        # Start prefetching thread
        self.prefetch_thread = threading.Thread(target=self.prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
        self.current_sample = 0
        return self
    
    def __next__(self):
        if self.current_sample >= self.num_samples:
            self.cleanup()
            raise StopIteration
        
        # Generate batch of data
        batch_captions = []
        batch_images = []
        
        # Try to get enough items for a batch
        timeout_per_item = 0.5  # seconds
        max_wait_time = self.batch_size * timeout_per_item
        start_time = time.time()
        
        while len(batch_images) < self.batch_size:
            # Check if we've waited too long
            if time.time() - start_time > max_wait_time:
                break
                
            try:
                img_tensor, caption = self.prefetch_queue.get(timeout=timeout_per_item)
                batch_images.append(img_tensor)
                batch_captions.append(caption)
            except Empty:
                # No items available in queue
                continue
        
        # Fill remaining slots with random data if needed
        while len(batch_images) < self.batch_size:
            random_img = torch.rand(self.image_channels, self.image_size[0], self.image_size[1])
            batch_images.append(random_img)
            batch_captions.append(random.choice(self.sample_captions))
            
        self.current_sample += 1
        return batch_captions, torch.stack(batch_images)
        
    def __len__(self):
        return self.num_samples
        
    def cleanup(self):
        """Clean up resources to prevent memory leaks"""
        if self.stop_event is not None:
            self.stop_event.set()
            
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1.0)
            
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
            
    def __del__(self):
        """Ensure resources are cleaned up when the object is garbage collected"""
        self.cleanup()

if __name__ == "__main__":
    import json
    import os
    from torchvision.utils import save_image
    
    # Create demo directory if it doesn't exist
    os.makedirs("demo", exist_ok=True)
    
    print("Loading and saving 10 images and captions...")
    
    # Initialize the dataloader with 10 samples
    loader = MultiModalDataLoader(batch_size=1, num_samples=10)
    
    sample_count = 0
    
    # Iterate through the dataloader
    for captions, images in loader:
        for i, (caption, image) in enumerate(zip(captions, images)):
            # Save the image as PNG
            image_path = f"demo/{sample_count}.png"
            save_image(image, image_path)
            
            # Save the caption as JSON
            caption_path = f"demo/{sample_count}.json"
            with open(caption_path, 'w') as f:
                json.dump({"caption": caption}, f)
            
            print(f"Saved image and caption {sample_count}: {caption[:30]}...")
            sample_count += 1
    
    print(f"Demo complete! Saved {sample_count} images and captions to demo/ directory")
