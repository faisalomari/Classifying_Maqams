from pydub import AudioSegment
import os

def resample_wav(input_file, output_file, target_sr=44100, target_bitrate='512k'):
    audio = AudioSegment.from_wav(input_file)
    audio = audio.set_frame_rate(target_sr)
    audio = audio.set_channels(1)  # Convert to mono if needed
    audio.export(output_file, format="wav", bitrate=target_bitrate)

def process_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            input_file = os.path.join(directory, filename)
            output_file = os.path.join(directory, f"resampled_{filename}")
            resample_wav(input_file, output_file)

# Replace 'path_to_directory' with the path to the directory containing the WAV files
path_to_directory = "/home/faisal/Desktop/datasetfull"
classes = ['Ajam']
for c in classes:
    process_files_in_directory(path_to_directory + "/" + c)