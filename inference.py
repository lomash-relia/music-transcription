
from guitar_notes_dataset import GuitarNotesDataset
import torch
from torch import nn
import torchaudio
from train import CNNnetwork

annotations_file = r"output.csv"
audio_dir = r"Notes Datasets"
sample_rate = 44100
num_samples = 44100

class_mapping = ['A', 'A-sharp', 'A-sharp 1', 'A-sharp 2', 'A1', 'A2', 'B', 'B1', 'B2', 'C', 'C-sharp', 'C-sharp 1', 'C-sharp 2', 'C1', 'C2', 'D', 'D-sharp', 'D-sharp 1', 'D-sharp 2', 'D1', 'D2', 'E', 'E1', 'E2', 'E3', 'F', 'F-sharp', 'F-sharp 1', 'F-sharp 2', 'F1', 'F2', 'G', 'G-sharp', 'G-sharp 1', 'G-sharp 2', 'G1', 'G2']


def predict(model: nn.Module, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == '__main__':
    
    cnn = CNNnetwork(37)
    state_dict = torch.load(r"guitarnet37_2.pth")
    cnn.load_state_dict(state_dict)
    
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
    for item in range(40,2000,55):
        input, target = guitar_dataset[item][0], guitar_dataset[item][1]
        input.unsqueeze_(0)
        
        # make inference
        predicted, expected = predict(cnn, input, target, class_mapping)
        print(f"Predicted: {predicted}, Expected: {expected}")