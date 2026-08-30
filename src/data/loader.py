import os
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomISLDataset(Dataset):
    def __init__(self, data_dir="data/processed", max_frames=60):
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.samples = []
        self.labels = []
        
        if os.path.exists(data_dir):
            self.class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            self.label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            
            for class_name in self.class_names:
                class_path = os.path.join(data_dir, class_name)
                for file_name in os.listdir(class_path):
                    if file_name.endswith(".npy"):
                        self.samples.append(os.path.join(class_path, file_name))
                        self.labels.append(self.label_to_idx[class_name])
        else:
            self.class_names = []
            self.label_to_idx = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]
        
        pose_data = np.load(file_path)  # Shape: [frames, 258]
        num_frames = pose_data.shape[0]
        
        # Pad or truncate to fixed max_frames length
        if num_frames < self.max_frames:
            padding = np.zeros((self.max_frames - num_frames, pose_data.shape[1]), dtype=np.float32)
            pose_data = np.vstack((pose_data, padding))
        else:
            pose_data = pose_data[:self.max_frames, :]
            
        tensor_data = torch.tensor(pose_data, dtype=torch.float32)
        return tensor_data, torch.tensor(label, dtype=torch.long)