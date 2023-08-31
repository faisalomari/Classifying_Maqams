import librosa
import numpy as np
import os
import pickle

def compute_mfcc_mean(audio_signal, sample_rate):
    # Compute the MFCCs
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate)

    # Compute the mean across MFCC coefficients for each time frame
    mfcc_mean = np.mean(mfccs, axis=1)

    return mfcc_mean


def compute_zcr_mean(audio_signal, sample_rate):
    # Compute the zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio_signal)

    # Compute the mean across the zero-crossing rate for each time frame
    zcr_mean = np.mean(zcr)

    return zcr_mean


def compute_chroma_mean(audio_signal, sample_rate, option=1):
    # Compute the chroma feature
    chroma = librosa.feature.chroma_stft(y=audio_signal, sr=sample_rate)
    # Compute the mean across the chroma feature for each time frame≈

    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean2 = np.mean(chroma_mean, axis=0)
    if option==1:
        return chroma_mean
    elif option==2:
        return chroma_mean2
    else:
        return chroma
    


def compute_rms_energy(audio_signal, sample_rate):
    # Compute the RMS energy
    rms_energy = librosa.feature.rms(y=audio_signal)

    # Get the mean RMS energy across all time frames
    mean_rms_energy = rms_energy.mean()

    return mean_rms_energy


def compute_spectral_centroid_mean(audio_signal, sample_rate):
    # Compute the spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(
        y=audio_signal, sr=sample_rate)

    # Get the mean spectral centroid across all time frames
    mean_spectral_centroid = spectral_centroids.mean()

    return mean_spectral_centroid


def compute_spectral_bandwidth_mean(audio_signal, sample_rate):
    # Compute the spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_signal, sr=sample_rate)

    # Get the mean spectral bandwidth across all time frames
    mean_spectral_bandwidth = spectral_bandwidth.mean()

    return mean_spectral_bandwidth


def compute_spectral_rolloff_mean(audio_signal, sample_rate):
    # Compute the spectral roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio_signal, sr=sample_rate)

    # Get the mean spectral roll-off across all time frames
    mean_spectral_rolloff = spectral_rolloff.mean()

    return mean_spectral_rolloff


def get_all_features(audio_file):
    audio_signal, sample_rate = librosa.load(audio_file, sr=None)
    mfcc_mean = compute_mfcc_mean(audio_signal, sample_rate)
    zcr_mean = compute_zcr_mean(audio_signal, sample_rate)
    chroma_mean = compute_chroma_mean(audio_signal, sample_rate, option=1)
    # chroma_mean2 = compute_chroma_mean(audio_signal, sample_rate, option=2)
    rms_energy = compute_rms_energy(audio_signal, sample_rate)
    spectral_centroid_mean = compute_spectral_centroid_mean(audio_signal, sample_rate)
    spectral_bandwidth_mean = compute_spectral_bandwidth_mean(audio_signal, sample_rate)
    spectral_rolloff_mean = compute_spectral_rolloff_mean(audio_signal, sample_rate)

    features = mfcc_mean
    features = np.append(features, zcr_mean)
    features = np.append(features, chroma_mean)
    # features = np.append(features, chroma_mean2)
    features = np.append(features, rms_energy)
    features = np.append(features, spectral_centroid_mean)
    features = np.append(features, spectral_bandwidth_mean)
    features = np.append(features, spectral_rolloff_mean)
    return features
def get_all_features2(audio_file):
    audio_signal, sample_rate = librosa.load(audio_file, sr=None)
    chroma = compute_chroma_mean(audio_signal, sample_rate, option=3)
    return chroma
def compute_features(audio_list, cache_file='features_cache.pkl', feature='mfcc'):
    if feature=='combined':
        if os.path.isfile(cache_file):
            print(f'Loading features from cache file: {cache_file}')
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f'Cache file not found. Computing features from scratch and saving to cache file: {cache_file}')
            features = []
            for i in range(len(audio_list)):
                audio_path, label_idx = audio_list[i]
                features.append(get_all_features(audio_path))
            # Save the computed features to the cache file
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            return features
    elif feature=='chroma':
        if os.path.isfile(cache_file):
            print(f'Loading features from cache file: {cache_file}')
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            print(f'Cache file not found. Computing features from scratch and saving to cache file: {cache_file}')
            features = []
            for i in range(len(audio_list)):
                audio_path, label_idx = audio_list[i]
                features.append(get_all_features2(audio_path))
            # Save the computed features to the cache file
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            return features

def custom_collate(batch):
    inputs, labels = zip(*batch)
    max_frames = max([m.shape[1] for m in inputs])
    padded_mfcc = []
    for m in inputs:
        pad_width = ((0, 0), (0, max_frames - m.shape[1]))
        padded_m = np.pad(m, pad_width=pad_width, mode='constant')
        padded_mfcc.append(padded_m)

    padded_mfcc = torch.from_numpy(np.array(padded_mfcc)).float()
    labels = torch.tensor(labels)
    return padded_mfcc, labels