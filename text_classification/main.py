import data_loader
import preprocessing
import model_training
from result_analysis import get_misclassified_examples, analyze_feature_importance

if __name__ == "__main__":
    print("Starting Text Classification Pipeline...\n")
    
    print("Step 1: Creating Labeled Dataset...")
    data_loader.create_labeled_dataset()
    
    print("Step 2: Preprocessing Data...")
    preprocessing.preprocess_text()
    
    print("Step 3: Training & Evaluating Models...")
    #model_training.train_and_evaluate_models()
    df, all_preds, all_true, vectorizer, trained_models = model_training.train_and_evaluate_models()
    
    print("Step 4: Analyzing Misclassifications & Feature Importance...")
    get_misclassified_examples(df, all_preds, all_true)
    analyze_feature_importance(vectorizer, trained_models)
    print("All steps completed")
