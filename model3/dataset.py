import os
import torchaudio
import torch
from torch.utils.data import Dataset

class MaqamDataset(Dataset):
    def __init__(self, mode='train', transform=None):
        self.mode = mode
        self.transform = transform
        self.data_dir = r"C:\Users\omari\Documents\GitHub\Maqam478"

        self.maqams = ['Ajam', 'Bayat', 'Hijaz', 'Kurd', 'Nahawand', 'Rast', 'Saba', 'Seka']
        self.audio_list = self._load_audio_list()

    def _load_audio_list(self):
        audio_list = []
        for i, maqam in enumerate(self.maqams):
            label_dir = os.path.join(self.data_dir, maqam)
            audio_list += [(os.path.join(label_dir, audio_name), i) for audio_name in os.listdir(label_dir) if audio_name.endswith('.wav')]
        return audio_list

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        audio_path, label_idx = self.audio_list[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform[0] # only keep the first channel
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label_idx
