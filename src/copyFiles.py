import os
import shutil

def find_next_number(output_folder):
    # Get a list of all files in the output folder
    files = os.listdir(output_folder)

    max_number = 0
    for file in files:
        # Split the file name at the "_" to get the class name and number separately
        parts = file.split("_")
        if len(parts) == 2:
            class_name, number_str = parts
            parts = number_str.split(".")
            number_str = parts[0]
            try:
                number = int(number_str)
                if number > max_number:
                    max_number = number
            except ValueError:
                pass
    return max_number + 1

def get_wav_files_in_directory(directory_path):
    wav_files = [file for file in os.listdir(directory_path) if file.endswith('.wav')]
    return wav_files

def copy_files_with_incrementing_numbers(input_folder, output_folder, classes, reader_name):
    for class_name in classes:
        class_files = [file for file in os.listdir(input_folder) if file.startswith(class_name)]
        for file in class_files:
            next_number = find_next_number(output_folder + "/" +file)
            ext = '.wav'
            source_path = os.path.join(input_folder, file)
            filess = get_wav_files_in_directory(source_path)
            for f in filess:
                new_name = f"{class_name}_{reader_name}_{next_number}{ext}"
                destination_path = os.path.join(output_folder, class_name)
                destination_path = os.path.join(destination_path, new_name)
                print(destination_path)
                src_path = os.path.join(source_path, f)
                shutil.copy(src_path, destination_path)
                next_number += 1

# Example usage:
reader_name = 'Saed_ghamdi'
input_folder = "/home/faisal/Desktop/MAQAMAT/mp3_maqamat/Readers/" + reader_name
output_folder = "/home/faisal/Desktop/datafullll"
classes = ['Ajam', 'Bayat', 'Hijaz', 'Kurd', 'Nahawand', 'Rast', 'Saba', 'Seka']

copy_files_with_incrementing_numbers(input_folder, output_folder, classes, reader_name)
