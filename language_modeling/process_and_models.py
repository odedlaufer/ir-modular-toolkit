import os
import re
from collections import Counter
import json
from nltk.stem import PorterStemmer

# פונקציות עיבוד טקסט
def load_stop_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(file.read().splitlines())

def remove_stop_words(tokens, stop_words):
    return [token for token in tokens if token not in stop_words]

def case_folding(tokens):
    return [token.lower() for token in tokens]

def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

# פונקציה ליצירת מודל לשוני
def create_language_model(tokens_folder, output_file):
    language_model = Counter()

    for filename in os.listdir(tokens_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(tokens_folder, filename), 'r', encoding='utf-8') as file:
                tokens = file.read().split()
                language_model.update(tokens)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(language_model.most_common(), file, ensure_ascii=False, indent=4)

    print(f"Language model saved to: {output_file}")

# פונקציה לעיבוד הקבצים וליצירת מודלים
def process_and_create_models(input_folder, output_folder, stop_words_file):
    stop_words = load_stop_words(stop_words_file)

    # תיקיות משנה לעיבוד
    step_a_folder = os.path.join(output_folder, 'step_a_remove_stop_words')
    step_b_folder = os.path.join(output_folder, 'step_b_case_folding')
    step_c_folder = os.path.join(output_folder, 'step_c_stemming')

    os.makedirs(step_a_folder, exist_ok=True)
    os.makedirs(step_b_folder, exist_ok=True)
    os.makedirs(step_c_folder, exist_ok=True)

    # שלב א': הסרת Stop Words
    for filename in os.listdir(input_folder):
        if filename.endswith('_tokens.txt'):
            input_path = os.path.join(input_folder, filename)
            with open(input_path, 'r', encoding='utf-8') as file:
                tokens = file.read().split()

            # הסרת Stop Words
            tokens_no_stop = remove_stop_words(tokens, stop_words)
            step_a_path = os.path.join(step_a_folder, filename)
            with open(step_a_path, 'w', encoding='utf-8') as file:
                file.write(' '.join(tokens_no_stop))

    # יצירת מודל לשוני לשלב א'
    create_language_model(step_a_folder, os.path.join(output_folder, 'model_after_stop_words.json'))

    # שלב ב': Case Folding
    for filename in os.listdir(step_a_folder):
        input_path = os.path.join(step_a_folder, filename)
        with open(input_path, 'r', encoding='utf-8') as file:
            tokens = file.read().split()

        # Case Folding
        tokens_case_folding = case_folding(tokens)
        step_b_path = os.path.join(step_b_folder, filename)
        with open(step_b_path, 'w', encoding='utf-8') as file:
            file.write(' '.join(tokens_case_folding))

    # יצירת מודל לשוני לשלב ב'
    create_language_model(step_b_folder, os.path.join(output_folder, 'model_after_case_folding.json'))

    # שלב ג': Stemming
    for filename in os.listdir(step_b_folder):
        input_path = os.path.join(step_b_folder, filename)
        with open(input_path, 'r', encoding='utf-8') as file:
            tokens = file.read().split()

        # Stemming
        tokens_stemmed = stemming(tokens)
        step_c_path = os.path.join(step_c_folder, filename)
        with open(step_c_path, 'w', encoding='utf-8') as file:
            file.write(' '.join(tokens_stemmed))

    # יצירת מודל לשוני לשלב ג'
    create_language_model(step_c_folder, os.path.join(output_folder, 'model_after_stemming.json'))

    print("All steps completed and language models created!")

# הגדרת נתיבים
input_folder = r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\Language modeling\tokenized'
output_folder = r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\Language modeling\processed'
stop_words_file = r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\stop_words.txt'

# הרצת התהליך
process_and_create_models(input_folder, output_folder, stop_words_file)
