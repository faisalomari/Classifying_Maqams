import os
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

class MaqamDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, max_length=7000):
        self.root_dir = root_dir
        self.file_list = []
        self.label_list = []
        self.label_to_index = {'Ajam':0, 'Bayat':1, 'Hijaz':2, 'Kurd':3, 'Rast':4, 'Saba':5, 'Nahawand':6, 'Seka':7}
        self.max_length = max_length

        # Store the unique class labels
        self.classes = list(self.label_to_index.keys())

        for label in os.listdir(root_dir):
            if label in self.label_to_index:
                label_index = self.label_to_index[label]
                label_dir = os.path.join(root_dir, label)
                for file in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file)
                    self.file_list.append(file_path)
                    self.label_list.append(label_index)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=1200,
            hop_length=600,
            f_min=50,
            f_max=8000,
            n_mels=64
        )(waveform)
        log_mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        # Convert log_mel_spectrogram to grayscale
        gray_log_mel_spectrogram = torch.mean(log_mel_spectrogram, dim=0, keepdim=True)
        
        # Pad spectrogram
        length = gray_log_mel_spectrogram.shape[2]
        if length < self.max_length:
            pad = torch.zeros((1, 64, self.max_length - length))
            gray_log_mel_spectrogram = torch.cat((gray_log_mel_spectrogram, pad), dim=2)
        elif length > self.max_length:
            gray_log_mel_spectrogram = gray_log_mel_spectrogram[:, :, :self.max_length]
        
        label = self.label_list[idx]
        return gray_log_mel_spectrogram, label
