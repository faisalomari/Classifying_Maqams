import tkinter as tk
import urllib.request
import json

def download_video():
    api_key = "YOUR_API_KEY_HERE"
    video_url = video_url_entry.get()
    video_id = video_url.split("=")[1]

    api_url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet,statistics"
    api_response = urllib.request.urlopen(api_url).read().decode()
    data = json.loads(api_response)['items'][0]['snippet']

    video_title = data['title']
    video_author = data['channelTitle']
    video_description = data['description']

    video_title_label.config(text=f"Title: {video_title}")
    video_author_label.config(text=f"Author: {video_author}")
    video_description_label.config(text=f"Description: {video_description}")

    stream_url = f"https://www.youtube.com/watch?v={video_id}"
    download_url_label.config(text=f"Download URL: {stream_url}")

root = tk.Tk()
root.geometry("500x300")
root.title("YouTube Downloader")

video_url_label = tk.Label(root, text="Enter the YouTube video URL:")
video_url_label.pack(pady=10)

video_url_entry = tk.Entry(root, width=50)
video_url_entry.pack(pady=5)

download_button = tk.Button(root, text="Download", command=download_video)
download_button.pack(pady=10)

video_title_label = tk.Label(root, text="")
video_title_label.pack(pady=5)

video_author_label = tk.Label(root, text="")
video_author_label.pack(pady=5)

video_description_label = tk.Label(root, text="")
video_description_label.pack(pady=5)

download_url_label = tk.Label(root, text="")
download_url_label.pack(pady=5)

root.mainloop()
