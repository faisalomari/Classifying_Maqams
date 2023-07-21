import librosa

# Load an audio file
audio_file = r"C:\Users\omari\Documents\GitHub\smalldataset\Kurd\Kurd_01.wav"
audio, sr = librosa.load(audio_file)

# Print the sampling rate
print(sr)
