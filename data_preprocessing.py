import pandas as pd
import re
import time

start = time.time()

#All functions for data preprocessing
def load_research_dataset(file_path: str) -> pd.DataFrame:
	"""
	Load a research dataset from a CSV file.

	Parameters:
	file_path:The path to the CSV

	Return
	A pandas DataFrame containing the loaded dataset.
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

# Tokenization function
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

def depollute_text(text: str, single_word_blacklist: set) -> str:
    if not isinstance(text, str):
        return text

    # Tokenize on whitespace (text is already normalized)
    tokens = text.split()

    # Remove blacklisted tokens
    tokens = [t for t in tokens if t not in single_word_blacklist]

    # Reconstruct text
    text = " ".join(tokens)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def tokenize_to_string(text: str) -> str:
    """
    Tokenize depolluted text and return a space-separated token string.
    Preserves stylistic punctuation: . , ! ?
    """
    if not isinstance(text, str):
        return ""

    punctuation_to_keep = {'.', ',', '!', '?'}

    tokens = []
    for word in text.split():
        if len(word) > 1 and word[-1] in punctuation_to_keep:
            tokens.append(word[:-1])
            tokens.append(word[-1])
        else:
            tokens.append(word)

    return " ".join(tokens)

#1 Data Prep 

#Loading different personality datasets
#Extravert Introvert csv
extrovert_introvert = load_research_dataset('data/extrovert_introvert.csv')
#Sensing Intuitive csv
sensing_intuitive = load_research_dataset('data/sensing_intuitive.csv')
# Feeling Thinking csv
feeling_thinking = load_research_dataset('data/feeling_thinking.csv')
# Judging Perceiving csv
judging_perceiving = load_research_dataset('data/judging_perceiving.csv')

#Additional datasets
	# #Age
	# age_data = load_research_dataset('data/age_data.csv')
	# #gender
	# gender_data = load_research_dataset('data/gender_data.csv')
	# nationality_data = load_research_dataset('data/nationality_data.csv')

#Summarize datasets
# summarize_dataset(extravert_introvert)


#Check NaN values 
print(extrovert_introvert.isna().sum())
print(sensing_intuitive.isna().sum())
print(feeling_thinking.isna().sum())
print(judging_perceiving.isna().sum())	

# 2 Basic text cleaning
#create a column post_clean to store cleaned posts
extrovert_introvert['post_clean'] = extrovert_introvert['post']
sensing_intuitive['post_clean'] = sensing_intuitive['post']
feeling_thinking['post_clean'] = feeling_thinking['post']
judging_perceiving['post_clean'] = judging_perceiving['post']


#text to lowercase
extrovert_introvert['post_clean'] = extrovert_introvert['post_clean'].str.lower()
sensing_intuitive['post_clean'] = sensing_intuitive['post_clean'].str.lower()
feeling_thinking['post_clean'] = feeling_thinking['post_clean'].str.lower()
judging_perceiving['post_clean'] = judging_perceiving['post_clean'].str.lower()

#print(judging_perceiving[['post']].head())

#Remove URLs
url_pattern = r'http\S+|www\S+|https\S+'

extrovert_introvert['post_clean'] = extrovert_introvert['post_clean'].str.replace(
	r'http\S+|www\.\S+',
	'',
	regex=True
)
sensing_intuitive['post_clean'] = sensing_intuitive['post_clean'].str.replace(
	r'http\S+|www\.\S+',
	'',
	regex=True
)
feeling_thinking['post_clean'] = feeling_thinking['post_clean'].str.replace(
	r'http\S+|www\.\S+',
	'',
	regex=True
)
judging_perceiving['post_clean'] = judging_perceiving['post_clean'].str.replace(
	r'http\S+|www\.\S+',
	'',
	regex=True
)

#Normalize punctuation
print("Before punctuation normalization:")
print(extrovert_introvert['post_clean'].head())
extrovert_introvert['post_clean'] = extrovert_introvert['post_clean'].apply(normalize_punctuation)
sensing_intuitive['post_clean'] = sensing_intuitive['post_clean'].apply(normalize_punctuation)
feeling_thinking['post_clean'] = feeling_thinking['post_clean'].apply(normalize_punctuation)
judging_perceiving['post_clean'] = judging_perceiving['post_clean'].apply(normalize_punctuation)

#Remove extra whitespace
extrovert_introvert['post_clean'] = extrovert_introvert['post_clean'].str.strip()
sensing_intuitive['post_clean'] = sensing_intuitive['post_clean'].str.strip()
feeling_thinking['post_clean'] = feeling_thinking['post_clean'].str.strip()
judging_perceiving['post_clean'] = judging_perceiving['post_clean'].str.strip()

#Tokenization
#I used rule-based tokenizer that separates sentence-final punctuation marks while preserving stylistic markers(exclamation and question marks).
extrovert_introvert['tokens'] = extrovert_introvert['post_clean'].apply(tokenize)
sensing_intuitive['tokens'] = sensing_intuitive['post_clean'].apply(tokenize)
feeling_thinking['tokens'] = feeling_thinking['post_clean'].apply(tokenize)
judging_perceiving['tokens'] = judging_perceiving['post_clean'].apply(tokenize)

#Check
print(extrovert_introvert[['post', 'post_clean', 'tokens']].head())


#3 Lexical depollution
#Create a column of depolluted posts
extrovert_introvert['post_depolluted'] = extrovert_introvert['post_clean']
sensing_intuitive['post_depolluted'] = sensing_intuitive['post_clean']	
feeling_thinking['post_depolluted'] = feeling_thinking['post_clean']
judging_perceiving['post_depolluted'] = judging_perceiving['post_clean']

#blacklist of MBTI-related terms

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

extrovert_introvert['post_depolluted'] = extrovert_introvert['tokens'].apply(
    lambda x: depollute_text(x, blacklist)
)
sensing_intuitive['post_depolluted'] = sensing_intuitive['tokens'].apply(
    lambda x: depollute_text(x, blacklist)
)
feeling_thinking['post_depolluted'] = feeling_thinking['tokens'].apply(
    lambda x: depollute_text(x, blacklist)
)
judging_perceiving['post_depolluted'] = judging_perceiving['tokens'].apply(
    lambda x: depollute_text(x, blacklist)
)

#Checks if there is leakege from blacklist
#Check if there is a blacklist word left
vocab = set(" ".join(extrovert_introvert['post_depolluted']).split())
print(vocab.intersection(blacklist))
vocab = set(" ".join(sensing_intuitive['post_depolluted']).split())
print(vocab.intersection(blacklist))
vocab = set(" ".join(feeling_thinking['post_depolluted']).split())
print(vocab.intersection(blacklist))
vocab = set(" ".join(judging_perceiving['post_depolluted']).split())
print(vocab.intersection(blacklist))	

#save cleaned datasets(all columns)
extrovert_introvert.to_csv('data/extrovert_introvert_cleaned.csv', index=False)
sensing_intuitive.to_csv('data/sensing_intuitive_cleaned.csv', index=False)	
feeling_thinking.to_csv('data/feeling_thinking_cleaned.csv', index=False)
judging_perceiving.to_csv('data/judging_perceiving_cleaned.csv', index=False)

end = time.time()
print(f"Runtime: {end - start:.2f} seconds")