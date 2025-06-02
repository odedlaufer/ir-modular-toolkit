import os
import chardet
import re

def detect_encoding(file_path):
    """
    Detect the encoding of a file using chardet.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def read_file_with_auto_encoding(file_path):
    """
    Read a file and decode it to utf-8, regardless of its original encoding.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'  # Use utf-8 as a fallback
    return raw_data.decode(encoding, errors='replace')

def process_files(input_folder, output_folder):
    """
    Process all .txt files in the input folder: tokenize and save to the output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.txt', '_tokens.txt'))

            # Read and decode the file
            text = read_file_with_auto_encoding(input_path)

            # Tokenize the text
            tokens = tokenize_text(text)

            # Save tokens to a new file
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(' '.join(tokens))
            print(f"Tokenized: {filename}")

def tokenize_text(text):
    """
    Tokenize text by splitting it on non-alphanumeric characters.
    """
    return [token for token in re.split(r'\W+', text) if token]

# Define paths
input_folder = r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\text'
output_folder = r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\Language modeling\tokenized'

# Run the process
process_files(input_folder, output_folder)
