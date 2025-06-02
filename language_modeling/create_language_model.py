import os
from collections import Counter
import json

def create_language_model(input_folder, output_file):
    """
    Create a language model (unigram) from tokenized files in the input folder.
    Save the results to a JSON file.
    """
    language_model = Counter()

    # Process all tokenized files
    for filename in os.listdir(input_folder):
        if filename.endswith('_tokens.txt'):
            input_path = os.path.join(input_folder, filename)

            # Read the content of the file
            with open(input_path, 'r', encoding='utf-8') as file:
                tokens = file.read().split()

            # Update the language model with tokens
            language_model.update(tokens)

    # Save the language model to a JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(language_model.most_common(), file, ensure_ascii=False, indent=4)

    print(f"Language model saved to: {output_file}")

# Define paths
input_folder = r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\Language modeling\tokenized'
output_file = r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\Language modeling\language_model.json'

# Run the process
create_language_model(input_folder, output_file)
