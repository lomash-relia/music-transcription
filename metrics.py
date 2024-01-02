from guitar_notes_dataset import GuitarNotesDataset
import torch
from torch import nn
import torchaudio
import numpy as np
from train import CNNnetwork
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassConfusionMatrix

annotations_file = r"output.csv"
audio_dir = r"Notes Datasets"
sample_rate = 44100
num_samples = 44100

class_mapping = ['A', 'A-sharp', 'A-sharp 1', 'A-sharp 2', 'A1', 'A2', 'B', 'B1', 'B2', 'C', 'C-sharp', 'C-sharp 1', 'C-sharp 2', 'C1', 'C2', 'D', 'D-sharp', 'D-sharp 1', 'D-sharp 2', 'D1', 'D2', 'E', 'E1', 'E2', 'E3', 'F', 'F-sharp', 'F-sharp 1', 'F-sharp 2', 'F1', 'F2', 'G', 'G-sharp', 'G-sharp 1', 'G-sharp 2', 'G1', 'G2']

def predict(model: nn.Module, input):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
    return predicted_index

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

    accuracy = MulticlassAccuracy(37)
    precision = MulticlassPrecision(37)
    recall = MulticlassRecall(37)
    confusion_matrix = MulticlassConfusionMatrix(37)

    for item in range(40, 1800, 55):
        input, target = guitar_dataset[item][0], guitar_dataset[item][1]
        input.unsqueeze_(0)

        # make inference
        predicted_index = predict(cnn, input)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

        # Convert target to torch.LongTensor
        target = torch.LongTensor([target])
        predicted_index = torch.LongTensor([predicted_index])

        # Update metrics
        accuracy(predicted_index, target)
        precision(predicted_index, target)
        recall(predicted_index, target)
        confusion_matrix(predicted_index, target)

        # print(f"Predicted: {predicted}, Expected: {expected}")

    # Compute and print final metrics
    final_accuracy = accuracy.compute()
    final_precision = precision.compute()
    final_recall = recall.compute()
    final_confusion_matrix = confusion_matrix.compute()

    print(f"Final Accuracy: {final_accuracy.item()}")
    print(f"Final Precision: {final_precision.item()}")
    print(f"Final Recall: {final_recall.item()}")
    print(f"Final Confusion Matrix:\n{final_confusion_matrix}")