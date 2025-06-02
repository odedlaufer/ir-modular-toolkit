# Information Retrieval Project

This repository contains a multi-module Information Retrieval project focused on real-world text processing tasks such as:

- **Text Clustering** of domain-specific corpora
- **Text Classification**
- **Language Modeling**

Each module demonstrates different approaches to handling and understanding textual data using classical and machine learning-based techniques.

## 📁 Project Structure

information_retrieval_project/
├── text_clustering/
│ ├── main.py
│ ├── clustering.py
│ ├── preprocess.py
│ ├── data_loader.py
│ ├── visualization.py
│ └── [3d_printing | filter_bubble | twitter_bias | etc.]
│
├── text_classification/
│ ├── main.py
│ ├── model_training.py
│ ├── result_analysis.py
│ ├── preprocessing.py
│ ├── data_loader.py
│ ├── config.py
│ └── labeled_dataset.csv
│
├── language_modeling/
│ ├── main.py
│ ├── check_encoding.py
│ ├── create_language_model.py
│ ├── compare_text_and_models.py
│ ├── stop_words.txt
│ └── remove_numbers.py

##  Getting Started

### Prerequisites
Ensure you have Python 3.8+ and install required packages:

```bash
pip install -r requirements.txt
```

### Run Example (Clustering)
```bash
cd text_clustering
python main.py
```

Each module may have its own `main.py` or script to run.

##  Features
- Preprocessing large collections of text
- Document clustering and visualization
- Classification using standard NLP techniques
- Modular design for easy experimentation

##  Built With
- `NLTK`, `NumPy`, `pandas`, `scikit-learn`, `matplotlib`
- Python standard libraries


