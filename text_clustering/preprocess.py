"""
preprocess.py
-------------
This module provides a custom text preprocessing function using NLTK and Python's string module.
It converts text to lowercase, tokenizes the text, and filters out stopwords and punctuation.

Functions:
    custom_preprocessor(text):
        Preprocesses a given text string by lowercasing, tokenizing, and filtering tokens.
"""

import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure that stopwords and punkt are downloaded:
# nltk.download('stopwords')
# nltk.download('punkt')

stop_words = set(stopwords.words('english'))
punctuation_set = set(string.punctuation)

def custom_preprocessor(text):
    """
    Preprocess the input text:
        - Lowercase the text.
        - Tokenize using NLTK's word_tokenize.
        - Remove tokens that are stopwords or punctuation.
    
    Parameters:
        text (str): The text to preprocess.
    
    Returns:
        str: The preprocessed text.
    """
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [
        token for token in tokens
        if token not in stop_words
           and token not in punctuation_set
           and not any(char.isdigit() for char in token)
    ]
    return " ".join(filtered_tokens)
