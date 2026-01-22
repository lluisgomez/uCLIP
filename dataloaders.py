import os
import json
from PIL import Image
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class Flickr30kDataLoader:
    def __init__(self, split="test", batch_size=32):
        self.split = split
        self.batch_size = batch_size
        self.dataset = None
        self.dataloader = None
    
    def load_data(self):
        # Load Flickr30k Dataset
        self.dataset = load_dataset("nlphuji/flickr30k")
        
        # Filter dataset based on the provided split
        self.dataset = self.dataset['test'].filter(lambda example: example['split'] == self.split)
        print(f"Filtered {self.split} set size: {len(self.dataset)}")
        
    def collate_fn(self, batch):
        images = [item["image"] for item in batch]
        captions = [item["caption"] for item in batch]
        return {"images": images, "captions": captions}
    
    def get_dataloader(self):
        if self.dataset is None:
            self.load_data()
        
        # Create DataLoader
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return self.dataloader


class MSCOCODataLoader:
    def __init__(self, split="test", batch_size=32):
        self.split = split
        self.batch_size = batch_size
        self.dataset = None
        self.dataloader = None

    def load_data(self):
        # Load MSCOCO Captions Dataset
        self.dataset = load_dataset("clip-benchmark/wds_mscoco_captions")

        # Access the required split
        if self.split in self.dataset:
            self.dataset = self.dataset[self.split]
        else:
            raise ValueError(f"Invalid split '{self.split}'. Available splits are: {list(self.dataset.keys())}")

        print(f"Loaded {self.split} set size: {len(self.dataset)}")

    def collate_fn(self, batch):
        images = [item["jpg"] for item in batch]
        captions = [item["txt"].split('\n') for item in batch]
        return {"images": images, "captions": captions}

    def get_dataloader(self):
        if self.dataset is None:
            self.load_data()

        # Create DataLoader (no shuffling needed for testing)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )
        return self.dataloader





# utility functions

def dataloader_with_indices(dataloader):
    start = 0
    for batch in dataloader:
        end = start + len(batch['images'])
        inds = torch.arange(start, end)
        yield batch['images'], batch['captions'], inds
        start = end

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

