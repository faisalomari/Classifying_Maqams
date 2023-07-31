import os
from pydub import AudioSegment

def cut_and_resample_wav(input_path, output_folder, file, segment_duration=30000, target_sample_rate=44100):
    # Load the input WAV file
    audio = AudioSegment.from_wav(input_path)

    # Check if the audio needs to be resampled
    if audio.frame_rate != target_sample_rate:
        audio = audio.set_frame_rate(target_sample_rate)

    # Calculate the segment duration in milliseconds
    segment_duration_ms = segment_duration

    # Get the total duration of the audio in milliseconds
    total_duration_ms = len(audio)

    # Calculate the number of segments needed
    num_segments = total_duration_ms // segment_duration_ms

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Cut and export each segment
    for i in range(30):
        start_time = i * segment_duration_ms
        end_time = (i + 1) * segment_duration_ms

        # Cut the segment
        segment = audio[start_time:end_time]

        # Export the segment to a WAV file
        output_file = os.path.join(output_folder, f"{file}_{i+600}.wav")
        segment.export(output_file, format="wav")

if __name__ == "__main__":
    # Replace "input_file.wav" with the path to your input WAV file
    files = ["ajam","bayat","hijaz","kurd","nahawand","nahawand2","saba","seka","seka_2"]
    folders = ["Ajam","Bayat","Hijaz","Kurd","Nahawand","Nahawand","Saba","Seka","Seka"]
    # files = ["nahawand2"]
    for i in range(9):
        print(i)
        file = files[i]
        folder = folders[i]
        input_file = "/home/faisal/Desktop/MAQAMAT/mp3_maqamat/Readers/afasi/wavFiles/"+file+".wav"

        # Replace "output_folder" with the path to the desired output folder
        output_folder = "/home/faisal/Desktop/MAQAMAT/fullqualitydataset/" + folder

        # Call the function to cut and resample the WAV file
        cut_and_resample_wav(input_file, output_folder,folder)

        print("Cutting and resampling completed.")
