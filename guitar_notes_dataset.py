import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from sklearn.preprocessing import LabelEncoder

audio_dir = r"D:\PyTorch project\Music Transcription\Notes Datasets"
sample_rate = 44100
num_samples = 44100

class GuitarNotesDataset(Dataset):

    def __init__(self, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.audio_dir = audio_dir
        self.df = self._get_dataframe_from_audio_dir(audio_dir)
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def _get_dataframe_from_audio_dir(self, audio_dir):
        file_paths = []
        labels = []

        for folder_name in os.listdir(audio_dir):
            folder_path = os.path.join(audio_dir, folder_name)
            
            # Check if it's a directory
            if os.path.isdir(folder_path):
                # Extract the sound name from the folder name
                _, sound_name = folder_name.split('. ', 1)
                
                # Iterate through each WAV file in the folder
                for wav_file in os.listdir(folder_path):
                    if wav_file.endswith('.wav'):
                        wav_file_path = os.path.join(folder_path, wav_file)
                        
                        # Append the file path and corresponding label to the lists
                        file_paths.append(wav_file_path)
                        labels.append(sound_name)

        # Create a dataframe from the lists
        df = pd.DataFrame({'File Path': file_paths, 'Label': labels})
        self.label_encoder = LabelEncoder()
        df['Class'] = self.label_encoder.fit_transform(df['Label'])
        # Save the entire DataFrame to a CSV file
        df.to_csv(r'D:\PyTorch project\Music Transcription\output.csv', index=False)
        return df

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.df)

    def __getitem__(self, index):
        # Get the path of the audio sample and its corresponding label
        audio_sample_path = self.df.iloc[index]['File Path']
        label = self.df.iloc[index]['Class']
        
        # Load the audio sample using torchaudio
        signal, samplerate = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        
        signal = self._resample_if_necessary(signal, samplerate)
        signal = self._mix_down_if_necessary(signal)        
        signal = self._right_pad_if_necessary(signal)        
        signal = self._cut_if_necessary(signal)        
        
        signal = self.transformation(signal)  # Apply spectrogram transformation
        # print(f"Signal shape:{signal.shape}")
        # Return the audio signal and its label
        # label = torch.tensor(self.label_encoder.transform([label]), dtype=torch.long).squeeze()

        return signal, label
    
    def _resample_if_necessary(self, signal, samplerate):
        if samplerate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(samplerate, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal =  signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:,:self.num_samples]
        return signal
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )
    
    gnd = GuitarNotesDataset(audio_dir,
                            mel_spectrogram,
                            sample_rate,
                            num_samples,    
                            device)
    
    print(f"There are {len(gnd)} notes in the dataset.")