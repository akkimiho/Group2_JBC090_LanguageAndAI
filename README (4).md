# Personality Prediction & Stylometric Analysis Framework

This repository contains a comprehensive research pipeline for predicting MBTI personality traits from text. The project focuses on distinguishing between **Content** (what is said) and **Style** (how it is said), with a specific emphasis on **Lexical Depollution** to remove bias and **Stylometric Feature Analysis**.

## 📋 Project Overview

The framework implements a machine learning pipeline to:
1.  **Preprocess** social media text (cleaning, tokenization).
2.  **Depollute** text by removing MBTI-specific keywords (e.g., "introvert", "ENTP") to prevent lexical leakage.
3.  **Extract Features**:
    *   **Content**: TF-IDF vectors (N-grams).
    *   **Stylometric**: Function words, POS ratios, lexical richness, syntactic markers, readability metrics.
4.  **Train Models**: Logistic Regression, SVM, Random Forest, LightGBM.
5.  **Evaluate**: Stratified Cross-validation (Macro-F1, Accuracy, AUC).
6.  **Analyze**: Compare raw vs. depolluted performance and integrate demographic data (RQ3).

## 🛠️ Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm optuna matplotlib seaborn nltk textstat
```

*Note: You may need to download NLTK data for the tokenizer and POS tagger:*
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## 📂 Project Structure

### Core Pipeline
*   **`test_optimized_data_processing.py`**: **(Start Here)** The primary preprocessing script. Cleans text, removes URLs, normalizes punctuation, and performs lexical depollution. Saves processed data to `data/clean/`.
*   **`main_experiment.py`**: The central orchestration script. Runs the full suite of experiments (Content, Stylometric, Depolluted, Demographics) and saves results as JSON.
*   **`visualize_results.py`**: Generates publication-ready visualizations (bar charts, heatmaps) from the experiment results.
*   **`generate_summary.py`**: Aggregates all JSON results into a single master CSV table.

### Utilities & Libraries
*   **`feature_extraction.py`**: Contains classes for feature engineering:
    *   `StylometricFeatures`: Extracts 20+ style markers (pronouns, hedges, readability, etc.).
    *   `ContentFeatures`: Standard TF-IDF vectorization.
*   **`model_trainer_file.py`**: Handles model training, evaluation, and Stratified K-Fold Cross-Validation.
*   **`hyperparameter_tuning.py`**: Implements Optuna for Bayesian optimization of model hyperparameters.
*   **`quick_start.py`**: A diagnostic script to verify data loading and pipeline functionality on a small subset.

### Legacy / Alternative Scripts
*   `data_preprocessing.py`: Basic preprocessing functions (alternative to the optimized script).
*   `data_preproccessing_2.py`: Script for checking class imbalances.

## 🚀 Usage Guide

### 1. Data Preparation
Ensure your raw CSV files (e.g., `extrovert_introvert.csv`) are located in `data/raw/`.
Run the optimized preprocessing script to clean, tokenize, and depollute the text.

```bash
python test_optimized_data_processing.py
```
*Output: Creates `data/clean/` containing `*_depolluted_text.csv` and `*_tokens_raw.csv`.*

### 2. Quick Start (Sanity Check)
Before committing to a long training run, verify that the data loads correctly and models can train.

```bash
python quick_start.py
```

### 3. Run Full Experiments
Execute the main pipeline. This will run experiments for all 4 MBTI tasks across multiple conditions (Content, Stylometric Raw, Stylometric Depolluted, Demographics) and perform hyperparameter tuning.

```bash
python main_experiment.py
```
*Output: Saves detailed JSON results in `results/` (e.g., `extrovert_introvert_content_cv.json`).*

### 4. Generate Visualizations
Create bar charts and heatmaps to analyze the hypotheses (Robustness, Content vs. Style, Demographic Gain).

```bash
python visualize_results.py
```
*Output: Saves images to `visuals/`.*

### 5. Summary Report
Generate a master CSV table of all results for easy comparison.

```bash
python generate_summary.py
```

## 🧠 Key Modules Detail

### Feature Extraction (`feature_extraction.py`)
The `StylometricFeatures` class extracts a dense vector of linguistic markers:
- **Function Words**: Pronouns, articles, prepositions (frequency).
- **POS Ratios**: Noun/Verb/Adjective/Adverb ratios.
- **Lexical Richness**: Type-Token Ratio (TTR), Hapax Legomena, average word length.
- **Syntactic**: Sentence length, punctuation usage.
- **Pragmatic**: Hedges, intensifiers, negations.
- **Readability**: Flesch-Kincaid, SMOG index.

### Model Training (`model_trainer_file.py`)
Supports the following algorithms with class-weight balancing:
- Logistic Regression (`logistic`)
- Support Vector Machines (`svm`)
- Random Forest (`rf`)
- LightGBM (`lgbm`)

## 🔬 Methodology Notes

- **Depollution**: The pipeline explicitly removes a "blacklist" of words (e.g., "INTJ", "introvert") to ensure the model learns linguistic patterns rather than relying on self-disclosure labels. This is critical for validating the "Stylometric Hypothesis".
- **RQ3 (Demographics)**: The `main_experiment.py` attempts to merge stylometric features with demographic data (Age, Gender, Nationality) if available in `data/raw/`.

## 📊 Hypotheses Tested

1.  **H1 (Robustness)**: Does removing explicit personality keywords significantly drop performance?
2.  **H2 (Content vs. Style)**: How do stylometric features compare to pure content (TF-IDF)?
3.  **RQ3**: Does adding demographic data improve prediction accuracy?