from pydub import AudioSegment
import os

# Set the desired length in milliseconds
desired_length = 30000
# Define a function to process the audio files
def process_audio_file(file_path):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Trim or pad the audio file to the desired length
    if len(audio) > desired_length:
        audio = audio[:desired_length]
    else:
        audio = audio + AudioSegment.silent(duration=desired_length - len(audio))

    # Export the processed audio file
    audio.export(file_path, format="wav")

# Process all the audio files in a directory
data_dir = r"C:\Users\omari\Documents\GitHub\smalldataset\Kurd"
for file_name in os.listdir(data_dir):
    if file_name.endswith(".wav"):
        file_path = os.path.join(data_dir, file_name)
        process_audio_file(file_path)
