import os
import soundfile as sf
import librosa
import wave

data_dir = r"C:\Users\USER\Documents\GitHub\newdataset_cutten"
output_dir = r"C:\Users\USER\Documents\GitHub\newdataset_cutten2"

# Set desired length in seconds
length = 10

# Loop through each maqam folder
for maqam in os.listdir(data_dir):
    maqam_dir = os.path.join(data_dir, maqam)
    if not os.path.isdir(maqam_dir):
        continue
    
    # Loop through each audio file in maqam folder
    for audio_file in os.listdir(maqam_dir):
        if not audio_file.endswith(".wav"):
            continue
        
        # Load audio file and resample to 22050 Hz if necessary
        audio_path = os.path.join(maqam_dir, audio_file)
        y, sr = sf.read(audio_path)

        y = librosa.resample(y, sr, 48000)
        print("sr = ", sr)
        # Get length in seconds and calculate number of samples to keep
        length_samples = int(length * sr)
        num_samples = min(len(y), length_samples)
        
        # Trim audio to desired length
        y_trimmed = y[:num_samples]
        
        # Export trimmed audio to output directory with same name
        output_path = os.path.join(output_dir, maqam, audio_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y_trimmed, sr, format='wav')
