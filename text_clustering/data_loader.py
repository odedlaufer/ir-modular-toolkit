"""
data_loader.py
--------------
This module handles loading text documents from a list of directory paths.
Each directory is assumed to represent a distinct class, and the folder name is used as the document's label.

Functions:
    load_documents(dir_paths):
        Reads all .txt files from the given directories and returns a pandas DataFrame
        with two columns: 'document' (the file content) and 'label' (the directory name).
"""

import os
import pandas as pd

def load_documents(dir_paths):
    """
    Load documents from the given directory paths.
    
    Parameters:
        dir_paths (list): A list of directory paths containing .txt files.
    
    Returns:
        DataFrame: A pandas DataFrame with columns 'document' and 'label'.
    """
    documents = []
    for dir_path in dir_paths:
        label = os.path.basename(dir_path.rstrip("/"))
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                documents.append({"document": content, "label": label})
    return pd.DataFrame(documents)
