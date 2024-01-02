from torch import torch,nn
from torch.utils.data import DataLoader
import torchaudio
from guitar_notes_dataset import GuitarNotesDataset
from cnn import CNNnetwork

annotations_file = "output.csv"
audio_dir = "Notes Datasets"
sample_rate = 44100
num_samples = 44100

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device).long()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    print(f"Training Loss: {average_loss}")

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print('-------')
    print("Training Completed")

if __name__ == '__main__':
    
    num_classes = 37
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )
    guitar_dataset = GuitarNotesDataset(
        audio_dir,
        mel_spectrogram,
        sample_rate,
        num_samples,
        device
    )

    train_data_loader = DataLoader(
        guitar_dataset,
        batch_size=3,
        shuffle=True
    )
    
    cnn = CNNnetwork(num_classes=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)
    
    train(cnn, train_data_loader, loss_fn, optimizer, device, epochs=50)

    torch.save(cnn.state_dict(), "guitarnet37_2.pth")
    print('Model Saved')
