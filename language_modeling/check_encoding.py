import os
import chardet

def detect_encoding(file_path):
    """
    Detect the encoding of a file using chardet.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def check_files_encoding(input_folder):
    """
    Check the encoding of all .txt files in a folder.
    """
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):  # Process only .txt files
            input_path = os.path.join(input_folder, filename)
            encoding = detect_encoding(input_path)
            print(f"{filename}: {encoding}")

# Define the path to the folder containing the text files
input_folder = r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\text'

# Run the encoding check
check_files_encoding(input_folder)
