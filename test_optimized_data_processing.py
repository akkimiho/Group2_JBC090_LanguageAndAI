import pandas as pd
import re
import time

start = time.time()
URL_PATTERN = r'http\S+|www\S+|https\S+'
#All functions for data preprocessing
def load_research_dataset(file_path: str) -> pd.DataFrame:
	"""
	Load a research dataset from a CSV file.
	"""
	try:
		data = pd.read_csv(file_path)
		return data
	except FileNotFoundError:
		print(f"Error: The file at {file_path} was not found.")
		return pd.DataFrame()
	except pd.errors.EmptyDataError:
		print("Error: The file is empty.")
		return pd.DataFrame()
	except pd.errors.ParserError:
		print("Error: There was a parsing error while reading the file.")
		return pd.DataFrame()
	
# Summary for  dataset
def summarize_dataset(data: pd.DataFrame) -> None:
	"""
	Print a summary of the dataset.
	"""
	if data.empty:
		print("The dataset is empty.")
		return

	print("Dataset Summary:")
	print(f"Number of rows: {data.shape[0]}")
	print(f"Number of columns: {data.shape[1]}")
	print("\nColumn Names:")
	print(data.columns.tolist())
	print("\nData Types:")
	print(data.dtypes)
	# print("\nMissing Values:")
	# print(data.isnull().sum())
	print("\nFirst 5 Rows:")
	print(data.head())

def df_clean(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Clean the DataFrame by removing NAN values.
	Returns:
	The cleaned DataFrame.
	"""
	#drop rows with any missing values)
	df = df.dropna()
	return df

def normalize_punctuation(text: str) -> str:
    """
    Normalize punctuation while preserving stylistic markers.
    Keeps . , ! ? ' and removes noisy symbols.
    Collapses repeated ! and ?.
    """
    if not isinstance(text, str):
        return text

    #Removing unwanted punctuation, but keep only once: . , ! ? ')
    text = re.sub(r"[^a-z0-9\s\.\,\!\?\']", "", text)

    #Normalize repeated punctuation
    text = re.sub(r'\!+', '!', text)
    text = re.sub(r'\?+', '?', text)

    #Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_text_series(s: pd.Series) -> pd.Series:

    s = s.astype(str).str.lower()
    # Remove URLs
    s = s.str.replace(URL_PATTERN, "", regex=True)

    # Normalize punctuation + whitespace
    s = s.apply(normalize_punctuation)

    return s


def depollute_text_single_words(text: str, blacklist: set) -> str:
    if not isinstance(text, str):
        return text
    tokens = text.split()
    tokens = [t for t in tokens if t not in blacklist]
    return " ".join(tokens)

def verify_no_leakage(text_series: pd.Series, blacklist: set) -> list:

    s = text_series.dropna()
    # s = s.sample(20000, random_state=42) #if data is too large, then we should do this
    vocab = set(" ".join(s).split())
    leaks = sorted(vocab.intersection(blacklist))
    return leaks

def tokenize(documents):
    """
    Tokenize text into words and punctuation tokens.
    Assumes text is already lowercased and punctuation-normalized.
    Preserves stylistic punctuation: . , ! ?
    """
    tokenized_documents = []

    punctuation_to_keep = {'.', ',', '!', '?'}

    for document in documents:
        tokens = []
        for word in document.split():
            # If word ends with punctuation we care about
            if word[-1] in punctuation_to_keep and len(word) > 1:
                tokens.append(word[:-1])
                tokens.append(word[-1])
            else:
                tokens.append(word)

        tokenized_documents.append(tokens)

    return tokenized_documents

#1 Data Prep 

#Loading different personality datasets
datasets = {
    "extrovert_introvert": ("data/extrovert_introvert.csv", "extrovert"),
    "sensing_intuitive":   ("data/sensing_intuitive.csv", "sensing"),
    "feeling_thinking":    ("data/feeling_thinking.csv", "feeling"), 
    "judging_perceiving":  ("data/judging_perceiving.csv", "judging"),
}

blacklist = [
     # mentions of type labels
    "intj", "intp", "entj", "entp", "infj", "infp", "enfj", "enfp",
    "istj", "istp", "estj", "estp", "isfj", "isfp", "esfj", "esfp",

    #other personality-related terms
    "introvert", "introverted", "introversion",
    "extrovert", "extroverted", "extravert", "extraverted",
    "sensor", "sensing",
    "intuitive", "intuition",
    "thinker", "feeler",
    "judger", "perceiver",

    # mbti jargon?
    "mbti", "myers-briggs", "16personalities",
    "cognitive functions",

]



for name, (path, label_col) in datasets.items():
    print(f"\n Processing: {name} => {path}")
    df = pd.read_csv(path)
    
    required_cols = {"auhtor_ID", "post", label_col}
    
    missing = required_cols - set(df.columns)
    
    if missing:
        raise ValueError(f"{name}: Missing columns: {missing}")

    # Drop missing critical fields ONLY
    df = df.dropna(subset=["auhtor_ID", "post", label_col])

    # Clean text (string)
    print(f"Cleaning text...")
    df["post_clean"] = clean_text_series(df["post"])
    

    # Depollution (single-word blacklist)
    print(f"Depollution of text...")
    df["post_depolluted"] = df["post_clean"].apply(
        lambda x: depollute_text_single_words(x, blacklist)
    )
	# Depollution
    df["post_depolluted"] = df["post_clean"].apply(
        lambda x: depollute_text_single_words(x, blacklist)
    )
    print("Verification...")
    
    # Verification
    leaks = verify_no_leakage(df["post_depolluted"], blacklist)
    print("Leakage count:", len(leaks))
    if leaks:
        print("Example leaks:", leaks[:20])
        

    # Class imbalance
    print("Imbalance checks...")
    
    counts = df[label_col].value_counts()
    perc = df[label_col].value_counts(normalize=True) * 100
    print("Class counts:\n", counts)
    print("Class %:\n", perc.round(2))

    # Tokenization
    print("Tokenization...")
    df["post_tokens_raw"] = df["post_clean"].apply(tokenize)
    df["post_tokens"] = df["post_depolluted"].apply(tokenize)

    # Save dataset with tokens without depollution
    out_tokens_raw = f"data/clean/{name}_tokens_raw.csv"
    df[["auhtor_ID", "post_tokens", label_col]].to_csv(out_tokens_raw, index=False)
    print("Saved:", out_tokens_raw)

    # Save string-based dataset (TF-IDF)out_text = f"data/clean/{name}_depolluted_text.csv"
    out_text = f"data/clean/{name}_depolluted_text.csv"
    df[["auhtor_ID", "post_depolluted", label_col]].to_csv(out_text, index=False)
    print("Saved:", out_text)

    # Save token-based dataset (stylometry) - depolluted
    out_tokens = f"data/clean/{name}_depolluted_tokens.csv"
    df[["auhtor_ID", "post_tokens", label_col]].to_csv(out_tokens, index=False)
    print("Saved:", out_tokens)

    # save full dataset with all columns for reference
    out_full = f"data/clean/{name}_full.csv"
    df.to_csv(out_full, index=False)
    print("Saved:", out_full)	


end = time.time()
print(f"Runtime: {end - start:.2f} seconds")