import os
import json
from collections import Counter

def load_text_files(folder_path):
    """
    Load all text files from a folder and return their content.
    """
    files_content = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                files_content.append(text)
    return files_content

def analyze_word_counts(files_content):
    """
    Analyze the total and unique word counts in the provided text files content.
    """
    all_words = []
    for content in files_content:
        all_words.extend(content.split())  # Split content into words

    total_words = len(all_words)
    unique_words = len(set(all_words))
    return total_words, unique_words

def compare_text_folders(text_folders):
    """
    Compare word counts across multiple text processing steps.
    """
    results = {}

    for description, folder_path in text_folders.items():
        files_content = load_text_files(folder_path)
        total_words, unique_words = analyze_word_counts(files_content)
        results[description] = {
            "total_words": total_words,
            "unique_words": unique_words
        }

    return results

def load_language_model(file_path):
    """
    Load a language model from a JSON file and return word statistics.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        language_model = json.load(file)
    total_words = sum(count for _, count in language_model)
    unique_words = len(language_model)
    return total_words, unique_words

def compare_language_models(models_folder):
    """
    Compare language models from different processing steps.
    """
    models = {
        "After Tokenization": os.path.join(models_folder, "language_model.json"),
        "After Stop Words Removal": os.path.join(models_folder, "model_after_stop_words.json"),
        "After Case Folding": os.path.join(models_folder, "model_after_case_folding.json"),
        "After Stemming": os.path.join(models_folder, "model_after_stemming.json")
    }

    results = {}

    for description, model_path in models.items():
        total_words, unique_words = load_language_model(model_path)
        results[description] = {
            "total_words": total_words,
            "unique_words": unique_words
        }

    return results

def save_results_to_file(results, title, file_path):
    """
    Save comparison results to a text file.
    """
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(f"\n{title}\n")
        file.write("-" * len(title) + "\n")
        for step, data in results.items():
            file.write(f"{step}:\n")
            file.write(f"  Total Words: {data['total_words']}\n")
            file.write(f"  Unique Words: {data['unique_words']}\n")
        file.write("\n")

# Define paths for text folders
text_folders = {
    "Tokenized Files": r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\Language modeling\tokenized',
    "After Stop Words Removal": r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\Language modeling\processed\step_a_remove_stop_words',
    "After Case Folding": r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\Language modeling\processed\step_b_case_folding',
    "After Stemming": r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\Language modeling\processed\step_c_stemming'
}

# Define the folder for language models
models_folder = r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\language models'

# Define the output file path
output_file = r'C:\Users\zohar\Desktop\סמסטר א\אחזור מידע\משימה מעשית\Language modeling\comparison_results.txt'

# Compare text folders
text_results = compare_text_folders(text_folders)
save_results_to_file(text_results, "Comparison of Text Folders (Word Counts)", output_file)

# Compare language models
model_results = compare_language_models(models_folder)
save_results_to_file(model_results, "Comparison of Language Models", output_file)
