import os
import re

def remove_numbers_from_text(text):
    """
    Remove whole numbers from the text using regular expressions.
    """
    return re.sub(r'\b\d+\b', '', text)  # Match only whole numbers as standalone words

def process_existing_files(input_folder):
    """
    Process all tokenized files in the input folder to remove numbers.
    """
    for filename in os.listdir(input_folder):
        if filename.endswith('_tokens.txt'):  # Process only tokenized files
            input_path = os.path.join(input_folder, filename)

            # Read the content of the file
            with open(input_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Remove numbers from the text
            updated_text = remove_numbers_from_text(text)

            # Save the updated text back to the file
            with open(input_path, 'w', encoding='utf-8') as file:
                file.write(updated_text)

            print(f"Processed (removed whole numbers): {filename}")

# Define the path to the folder containing the tokenized files
input_folder = r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\Language modeling\tokenized'

# Run the process
process_existing_files(input_folder)
