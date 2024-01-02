from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torchaudio
from torch.utils.data import DataLoader
from torchvision import transforms
from guitar_notes_dataset import GuitarNotesDataset
from cnn import CNNnetwork

app = FastAPI()

# Load the trained model
class_mapping = ['A', 'A-sharp', 'A-sharp 1', 'A-sharp 2', 'A1', 'A2', 'B', 'B1', 'B2', 'C', 'C-sharp', 'C-sharp 1', 'C-sharp 2', 'C1', 'C2', 'D', 'D-sharp', 'D-sharp 1', 'D-sharp 2', 'D1', 'D2', 'E', 'E1', 'E2', 'E3', 'F', 'F-sharp', 'F-sharp 1', 'F-sharp 2', 'F1', 'F2', 'G', 'G-sharp', 'G-sharp 1', 'G-sharp 2', 'G1', 'G2']
cnn = CNNnetwork(37)
state_dict = torch.load("guitarnet37_2.pth")
cnn.load_state_dict(state_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the mel spectrogram transformation
mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100,
    n_fft=1024,
    hop_length=512,
    n_mels=64,
)

# Create the GuitarNotesDataset instance
audio_dir = "Notes Datasets"
sample_rate = 44100
num_samples = 44100
guitar_dataset = GuitarNotesDataset(
    audio_dir,
    mel_spectrogram_transform,
    sample_rate,
    num_samples,
    device
)

# Define the prediction response model
class SoundPrediction(BaseModel):
    sound_class: str

# Preprocess the audio input
def preprocess_audio(audio_file_path, target_sample_rate=44100, num_samples=44100):
    signal, _ = torchaudio.load(audio_file_path)
    signal = signal.mean(dim=0, keepdim=True) if signal.shape[0] > 1 else signal
    signal = transforms.Pad((0, num_samples - signal.shape[1]))(signal)
    
    # Ensure the input has 1 channel
    if signal.shape[0] != 1:
        signal = signal[:1, :]
    
    signal = mel_spectrogram_transform(signal)
    signal = signal.unsqueeze(0)
    return signal

# Define the prediction route
@app.post("/predict_sound", response_model=SoundPrediction)
async def predict_sound(file: UploadFile = UploadFile(...)):
    try:
        if file.file is None:
            raise HTTPException(status_code=400, detail="Invalid file: None")

        # Preprocess the uploaded audio file
        preprocessed_input = preprocess_audio(file.file)

        # Make predictions with the model
        with torch.no_grad():
            input_data = preprocessed_input.to(device)
            predicted_index = cnn(input_data).argmax(1).item()
            predicted = class_mapping[predicted_index]

        return JSONResponse(content={"sound_class": predicted})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
