import os
import torch
import torchaudio

class MaqamDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, sample_rate=22050):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.classes, self.class_to_idx = self.find_classes(self.data_dir)
        self.filenames, self.labels = self.load_data(self.data_dir, self.class_to_idx, self.sample_rate)

    def __getitem__(self, index):
        file_path = self.filenames[index]
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform.mean(0, keepdim=True)
        waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)
        waveform = waveform.squeeze(0)

        label = self.labels[index]
        return waveform, label

    def __len__(self):
        return len(self.filenames)

    def find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def load_data(self, dir, class_to_idx, sample_rate):
        labels = []
        filenames = []
        for target_class in os.listdir(dir):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(dir, target_class)
            for filename in os.listdir(target_dir):
                filenames.append(os.path.join(target_dir, filename))
                labels.append(class_index)

        # Pad all waveforms to the same length
        waveforms = []
        for filename in filenames:
            waveform, sample_rate = torchaudio.load(filename)
            waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)
            waveform = waveform.squeeze(0)
            waveforms.append(waveform)
        waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        
        return waveforms, torch.tensor(labels)
