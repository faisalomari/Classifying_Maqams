import librosa
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from matplotlib import cm

def cut_and_export_audio(input_audio_path, start_time, end_time, output_audio_path):
    # Load the audio signal
    audio_signal, sample_rate = librosa.load(input_audio_path, sr=None)

    # Calculate the start and end samples based on the time in seconds
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Extract the audio segment
    audio_segment = audio_signal[start_sample:end_sample]

    # Export the audio segment as a new WAV file using soundfile
    sf.write(output_audio_path, audio_segment, samplerate=sample_rate)

def compute_and_save_mfcc(audio_path, output_path):
    # Load the audio signal
    audio_signal, sample_rate = librosa.load(audio_path, sr=None)

    # Compute the MFCCs
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate)

    # Save MFCC plot as PNG with 'hot' colormap
    plt.figure(figsize=(8, 6))
    plt.imshow(mfccs, aspect='auto', origin='lower', cmap='hot')
    plt.colorbar()
    plt.title('MFCC')
    plt.xlabel('Time Frame')
    plt.ylabel('MFCC Coefficient')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the MFCC plot
    plt.savefig(output_path)
    plt.close()
def compute_and_save_mfcc2(audio_path, output_path):
    (rate, sig) = wav.read(audio_path)
    mfcc_feat = mfcc(sig, rate)

    fig, ax = plt.subplots()
    mfcc_data = np.swapaxes(mfcc_feat, 0, 1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap='coolwarm', origin='lower', aspect='auto')
    ax.set_title('MFCC')
    plt.colorbar(cax)  # Add colorbar

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the MFCC plot
    plt.savefig(output_path)
    plt.close()

def compute_and_save_chroma(audio_path, output_path):
    # Load the audio signal
    audio_signal, sample_rate = librosa.load(audio_path, sr=None)

    # Compute the chroma feature
    chroma = librosa.feature.chroma_stft(y=audio_signal, sr=sample_rate)

    # Save chroma plot as PNG with 'coolwarm' colormap
    plt.figure(figsize=(8, 6))
    plt.imshow(chroma, aspect='auto', origin='lower', cmap='coolwarm')
    plt.colorbar()
    plt.title('Chroma')
    plt.xlabel('Time Frame')
    plt.ylabel('Chroma Coefficient')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the chroma plot
    plt.savefig(output_path)
    plt.close()

def compute_and_save_rms(audio_path, output_path):
    # Load the audio signal
    audio_signal, sample_rate = librosa.load(audio_path, sr=None)

    # Compute the RMS energy
    rms_energy = librosa.feature.rms(y=audio_signal)

    # Get the RMS energy as a 1D array (signal waveform)
    rms_signal = rms_energy[0]

    # Create a time axis for the RMS signal
    time_axis = np.arange(len(rms_signal)) * (len(audio_signal) / len(rms_signal)) / sample_rate

    # Plot the RMS energy as a signal waveform
    plt.figure(figsize=(8, 6))
    plt.plot(time_axis, rms_signal)
    plt.title('RMS Energy')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS Energy')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the RMS energy plot
    plt.savefig(output_path)
    plt.close()

def compute_and_save_spectral_centroid(audio_path, output_path):
    # Load the audio signal
    audio_signal, sample_rate = librosa.load(audio_path, sr=None)

    # Compute the spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_signal, sr=sample_rate)

    # Get the spectral centroid as a 1D array (signal waveform)
    spectral_centroid_signal = spectral_centroids[0]

    # Create a time axis for the spectral centroid signal
    time_axis = np.arange(len(spectral_centroid_signal)) * (len(audio_signal) / len(spectral_centroid_signal)) / sample_rate

    # Plot the spectral centroid as a signal waveform
    plt.figure(figsize=(8, 6))
    plt.plot(time_axis, spectral_centroid_signal)
    plt.title('Spectral Centroid')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spectral Centroid')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the spectral centroid plot
    plt.savefig(output_path)
    plt.close()

def compute_and_save_spectral_bandwidth(audio_path, output_path):
    # Load the audio signal
    audio_signal, sample_rate = librosa.load(audio_path, sr=None)

    # Compute the spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_signal, sr=sample_rate)

    # Get the spectral bandwidth as a 1D array (signal waveform)
    spectral_bandwidth_signal = spectral_bandwidth[0]

    # Create a time axis for the spectral bandwidth signal
    time_axis = np.arange(len(spectral_bandwidth_signal)) * (len(audio_signal) / len(spectral_bandwidth_signal)) / sample_rate

    # Plot the spectral bandwidth as a signal waveform
    plt.figure(figsize=(8, 6))
    plt.plot(time_axis, spectral_bandwidth_signal)
    plt.title('Spectral Bandwidth')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spectral Bandwidth')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the spectral bandwidth plot
    plt.savefig(output_path)
    plt.close()

def compute_and_save_spectral_rolloff(audio_path, output_path):
    # Load the audio signal
    audio_signal, sample_rate = librosa.load(audio_path, sr=None)

    # Compute the spectral roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sample_rate)

    # Get the spectral roll-off as a 1D array (signal waveform)
    spectral_rolloff_signal = spectral_rolloff[0]

    # Create a time axis for the spectral roll-off signal
    time_axis = np.arange(len(spectral_rolloff_signal)) * (len(audio_signal) / len(spectral_rolloff_signal)) / sample_rate

    # Plot the spectral roll-off as a signal waveform
    plt.figure(figsize=(8, 6))
    plt.plot(time_axis, spectral_rolloff_signal)
    plt.title('Spectral Rolloff')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Spectral Rolloff')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the spectral roll-off plot
    plt.savefig(output_path)
    plt.close()

def compute_and_save_zcr(audio_path, output_path):
    # Load the audio signal
    audio_signal, sample_rate = librosa.load(audio_path, sr=None)

    # Compute the zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio_signal)

    # Get the zero-crossing rate as a 1D array (signal waveform)
    zcr_signal = zcr[0]

    # Create a time axis for the zero-crossing rate signal
    time_axis = np.arange(len(zcr_signal)) * (len(audio_signal) / len(zcr_signal)) / sample_rate

    # Plot the zero-crossing rate as a signal waveform
    plt.figure(figsize=(8, 6))
    plt.plot(time_axis, zcr_signal)
    plt.title('Zero Crossing Rate')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Zero Crossing Rate')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the zero-crossing rate plot
    plt.savefig(output_path)
    plt.close()

# Seka_55 Saba_58 Rast_67 Nahawand_56 Kurd_51 Hijaz_70 Bayat_57 ajam
audio_path = "/home/faisal/Desktop/figures_data/ajam.wav"
cut_and_export_audio(audio_path, 0, 10, audio_path)
audio_signal, sample_rate = librosa.load(audio_path, sr=None)

# Save the plots as PNG images
output_path = 'results/background/zrc.png'
compute_and_save_zcr(audio_path, output_path)