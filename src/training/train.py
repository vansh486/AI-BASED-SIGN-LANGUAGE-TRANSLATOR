import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.loader import CustomISLDataset  # or your specific data loader file
from src.models.hybrid import HybridModel   # Update with your hybrid class name if different

def train_model():
    dataset = CustomISLDataset(data_dir="data/processed")
    if len(dataset) == 0:
        print("No processed tensors found! Check your preprocessed folders.")
        return
        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize the model (adjust num_classes to match your saved sign folders)
    num_classes = len(dataset.class_names) if hasattr(dataset, 'class_names') else 5
    model = HybridModel(num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training on {len(dataset)} samples across {num_classes} classes...")
    model.train()
    
    for epoch in range(15):  # Train for 15 epochs
        total_loss = 0
        for tensors, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/15 | Loss: {total_loss / len(dataloader):.4f}")

    # Ensure models directory exists and save weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/isl_hybrid_model.pth")
    print("Training complete! Model saved to models/isl_hybrid_model.pth")

if __name__ == "__main__":
    train_model()