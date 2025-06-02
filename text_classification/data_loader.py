"""
data_loader.py

This module is responsible for:
- Scanning directories containing text files.
- Assigning labels based on the source directory.
- Writing the labeled dataset to a CSV file.

Functions:
    - create_labeled_dataset(): Scans directories and saves labeled text data.

Usage:
    Run this script directly or import `create_labeled_dataset()` in `main.py`.

Example:
    from data_loader import create_labeled_dataset
    create_labeled_dataset()
"""

import os
import csv
from config import DIR_TARGET, DIR_RANDOM, OUTPUT_CSV

def create_labeled_dataset():
    """
    Scans directories for text files, assigns labels, and writes them to a CSV file.
    - Files from `DIR_TARGET` are labeled as 1 (category documents).
    - Files from `DIR_RANDOM` are labeled as 0 (non-category documents).

    Output:
        A labeled dataset saved to OUTPUT_CSV.
    """
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8', errors='ignore') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "text", "label"])

        for directory, label in [(DIR_TARGET, 1), (DIR_RANDOM, 0)]:
            for filename in os.listdir(directory):
                if filename.endswith('.txt'):
                    full_path = os.path.join(directory, filename)
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        text = infile.read()
                    text = text.replace('\n', ' ').replace('\r', ' ').strip()
                    writer.writerow([filename, text, label])

    print(f"Label dataset saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    create_labeled_dataset()
