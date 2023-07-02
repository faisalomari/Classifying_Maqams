import wave
import os
import tkinter as tk
from tkinter import filedialog

# create a root window for the file dialog box
root = tk.Tk()
root.withdraw()

# open file dialog box to choose WAV file
file_path = filedialog.askopenfilename(
    title="Select WAV File",
    filetypes=[("WAV Files", "*.wav")]
)

# ask user to specify cut length in seconds
cut_length = 10

# open WAV file for reading
with wave.open(file_path, 'rb') as wave_read:

    # get sample rate and number of channels from WAV file
    sample_rate = wave_read.getframerate()
    num_channels = wave_read.getnchannels()

    # calculate number of frames to read for the desired cut length
    cut_frames = int(cut_length * sample_rate)

    # read frames from WAV file
    frames = wave_read.readframes(cut_frames)

# create new WAV file for writing the cut
cut_file = os.path.splitext(file_path)[0] + ".wav"
with wave.open(cut_file, 'wb') as wave_write:

    # set parameters for new WAV file
    wave_write.setnchannels(num_channels)
    wave_write.setsampwidth(wave_read.getsampwidth())
    wave_write.setframerate(sample_rate)

    # write frames to new WAV file
    wave_write.writeframes(frames)

print(f"The cut has been extracted to {cut_file}.")
