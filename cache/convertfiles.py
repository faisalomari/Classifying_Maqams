from pydub import AudioSegment
import os

def mp3_to_wav(input_path, output_path, duration=30):
    # Load the MP3 audio file
    audio = AudioSegment.from_mp3(input_path)

    # Ensure the audio is in mono channel (optional)
    audio = audio.set_channels(1)

    # Set the output format to WAV
    audio = audio.set_frame_rate(44100)
    audio = audio.set_sample_width(2)

    # Calculate the duration in milliseconds
    duration_ms = duration * 10000000

    # If the audio is shorter than the desired duration, keep the original length
    if len(audio) < duration_ms:
        duration_ms = len(audio)

    # Cut the audio to the desired duration
    audio = audio[:duration_ms]

    # Export the audio to WAV format
    audio.export(output_path, format="wav")

if __name__ == "__main__":
    # Replace "input_file.mp3" with the path to your input MP3 file
    input_file = "/home/faisal/Desktop/MAQAMAT/mp3_maqamat/Readers/Saed_ghamdi/rast_1"

    # Replace "output_file.wav" with the path to the output WAV file
    output_file = input_file + ".wav"
    input_file = input_file + ".mp3"
    # Convert and cut the audio to 30 seconds
    mp3_to_wav(input_file, output_file, duration=30)

    print("Conversion and cutting completed.")
