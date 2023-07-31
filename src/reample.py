import os
import librosa
import soundfile as sf

def convert_to_44100hz(input_folder, output_folder):
    # Get a list of all WAV files in the input folder
    wav_files = [file for file in os.listdir(input_folder) if file.endswith('.wav')]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for file in wav_files:
        # Load the audio file
        audio_path = os.path.join(input_folder, file)
        y, sr = librosa.load(audio_path, sr=None)

        # Resample to 44.1kHz (44100Hz)
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=44100)

        # Export the resampled audio to the output folder
        output_path = os.path.join(output_folder, file)
        sf.write(output_path, y_resampled, 44100)

        print(f"{file} converted and saved to {output_path}")

# Example usage:
m = ['Ajam', 'Bayat', 'Hijaz', 'Kurd', 'Nahawand', 'Rast', 'Saba', 'Seka']
for k in m:
    input_folder = "/home/faisal/Desktop/MAQAMAT/Maqam478/Dataset2/" + k
    output_folder = "/home/faisal/Desktop/datasetfull/" + k
    convert_to_44100hz(input_folder, output_folder)
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("Conversion FINISHED!")