import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import PhysicsMLP

EPOCHS = 100
BATCH_SIZE = 300
LEARNING_RATE = 1e-3


def load_tensors():
    """
    Loads the saved numpy arrays from prepare_data.py .
    """
    train_states = torch.tensor(np.load("train_states.npy"), dtype=torch.float32)
    train_deltas = torch.tensor(np.load("train_deltas.npy"), dtype=torch.float32)
    val_states = torch.tensor(np.load("val_states.npy"), dtype=torch.float32)
    val_deltas = torch.tensor(np.load("val_deltas.npy"), dtype=torch.float32)

    return train_states, train_deltas, val_states, val_deltas


def train():
    
    train_states, train_deltas, val_states, val_deltas = load_tensors()

    # loading the dataset and shuffling
    train_dataset = TensorDataset(train_states, train_deltas)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)

    model = PhysicsMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0


        for batch_states, batch_deltas in train_loader:
            optimizer.zero_grad()
            predicted = model(batch_states)
            mse_loss = loss_fn(predicted, batch_deltas)
            mse_loss.backward()
            optimizer.step()
            total_train_loss += mse_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_states)
            val_loss = loss_fn(val_predictions, val_deltas).item()



        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    torch.save(model.state_dict(), "physics_mlp.pt")
    print("Model saved to physics_mlp.pt")


if __name__ == "__main__":
    train()
