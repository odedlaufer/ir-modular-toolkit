"""
preprocessing.py

This module preprocesses text data by:
- Removing null bytes.
- Converting text to lowercase.
- Tokenizing text into words.
- Removing stop words and punctuation.
- Saving processed text to a CSV file.

Functions:
    - preprocess_text(): Reads raw data and applies preprocessing steps.

Usage:
    Run this script directly or import `preprocess_text()` in `main.py`.

Example:
    from preprocessing import preprocess_text
    preprocess_text()
"""

import csv
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from config import OUTPUT_CSV, PROCESSED_CSV

# Load stop words and punctuation set
stop_words = set(stopwords.words('english'))
punctuation_set = set(string.punctuation)

def remove_null_generator(file_obj):
    """
    Generator function to remove null bytes from a file stream.

    Args:
        file_obj: A file object to read lines from.

    Yields:
        Lines with null bytes removed.
    """
    for line in file_obj:
        yield line.replace('\x00', '')

def preprocess_text():
    """
    Reads raw labeled data, preprocesses text, and writes it to a new CSV file.

    Steps:
        1. Converts text to lowercase.
        2. Tokenizes text into words.
        3. Removes stop words and punctuation.
        4. Saves cleaned text to PROCESSED_CSV.

    Output:
        A processed dataset saved to PROCESSED_CSV.
    """
    with open(OUTPUT_CSV, 'r', encoding='utf-8', errors='replace') as infile_raw, \
         open(PROCESSED_CSV, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(remove_null_generator(infile_raw))
        writer = csv.DictWriter(outfile, fieldnames=["filename", "processed_text", "label"])
        writer.writeheader()

        for row in reader:
            text = row["text"].lower()  
            tokens = word_tokenize(text)  
            filtered_tokens = [token for token in tokens if token not in stop_words and token not in punctuation_set]
            processed_text = " ".join(filtered_tokens)

            writer.writerow({"filename": row["filename"], "processed_text": processed_text, "label": row["label"]})

    print(f"Preprocessing complete. Results saved to: {PROCESSED_CSV}")

if __name__ == "__main__":
    preprocess_text()
