import os
import torchaudio
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from features_extractor import*

class MaqamDataset(Dataset):
    def __init__(self, mode='train', transform=None, cache_file='running_cache/cache.pkl', feature='mfcc'):
        self.data_dir = "/home/faisal/Desktop/datasetfull"

        self.mode = mode
        self.transform = transform
        test_val_size = 0.3
        self.feature = feature
        
        self.maqams = ['Ajam', 'Bayat', 'Hijaz', 'Kurd', 'Nahawand', 'Rast', 'Saba', 'Seka']
        self.audio_list = self._load_audio_list()
        if mode == 'train':
            train_list, val_list = train_test_split(self.audio_list, test_size=test_val_size, random_state=42, stratify=[label for (_, label) in self.audio_list])
            self.audio_list = train_list
            self.features = compute_features(self.audio_list, cache_file="running_cache/train_features.pkl", feature=feature)
        elif mode == 'val':
            train_list, set2 = train_test_split(self.audio_list, test_size=test_val_size, random_state=42, stratify=[label for (_, label) in self.audio_list])
            val_set, test_set = train_test_split(set2, test_size=0.5, random_state=42, stratify=[label for (_, label) in set2])
            self.audio_list = val_set
            self.features = compute_features(self.audio_list, cache_file="running_cache/val_features.pkl", feature=feature)
        elif mode == 'test':
            train_list, set2 = train_test_split(self.audio_list, test_size=test_val_size, random_state=42, stratify=[label for (_, label) in self.audio_list])
            val_set, test_set = train_test_split(set2, test_size=0.5, random_state=42, stratify=[label for (_, label) in set2])
            self.audio_list = test_set
            self.features = compute_features(self.audio_list, cache_file="running_cache/test_features.pkl", feature=feature)

        self.cache_file = cache_file
        self.data = self._load_data_from_cache_or_compute()

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
        if(self.feature == 'mfcc'):
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform[0] # only keep the first channel
            if self.transform:
                waveform = self.transform(waveform)
            mfcc = self.compute_mfcc(waveform).T
            mfcc = torch.from_numpy(mfcc).float()
            return mfcc, label_idx
        elif(self.feature == 'combined'):
            features = self.features[idx]
            return features, label_idx
        elif(self.feature == 'chroma'):
            features = self.features[idx]
            return features, label_idx
  
    def pad_to_max_length(self, max_length):
        for i in range(len(self)):
            padded_data = F.pad(self.data[i][0], (0, max_length - len(self.data[i][0])), 'constant', 0)
            self.data[i] = (padded_data, self.data[i][1])

    def compute_mfcc(self, waveform):
        # Compute the MFCC of the waveform
        n_fft = 1024*16
        hop_length = 256*8
        n_mels = 100
        sr = 48000
        n_mfcc = 20
        waveform = waveform.numpy()  # Convert PyTorch tensor to NumPy array
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr)
        mfcc = np.transpose(mfcc)
        mfcc = mfcc.astype(np.float32)  # Ensure data type is compatible with np.issubdtype()
        return mfcc
    
    def _load_data_from_cache_or_compute(self):
        if os.path.isfile(self.cache_file):
            print(f'Loading data from cache file: {self.cache_file}')
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f'Cache file not found. Computing data from scratch and saving to cache file: {self.cache_file}')
            data = [self.__getitem__(i) for i in range(len(self))]
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            return data
