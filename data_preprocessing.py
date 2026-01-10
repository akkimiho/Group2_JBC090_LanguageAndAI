import pandas as pd
import re

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

#1 Data Prep 

#Loading different personality datasets
#Extravert Introvert csv
extravert_introvert = load_research_dataset('data/extrovert_introvert.csv')
#Sensing Intuitive csv
sensing_intuitive = load_research_dataset('data/sensing_intuitive.csv')
# Feeling Thinking csv
feeling_thinking = load_research_dataset('data/feeling_thinking.csv')
# Judging Perceiving csv
judging_perceiving = load_research_dataset('data/judging_perceiving.csv')

#Summarize datasets
# summarize_dataset(extravert_introvert)


#Check NaN values 
print(extravert_introvert.isna().sum())
print(sensing_intuitive.isna().sum())
print(feeling_thinking.isna().sum())
print(judging_perceiving.isna().sum())	

# 2 Basic text cleaning
#text to lowercase
extravert_introvert['post'] = extravert_introvert['post'].str.lower()
sensing_intuitive['post'] = sensing_intuitive['post'].str.lower()
feeling_thinking['post'] = feeling_thinking['post'].str.lower()
judging_perceiving['post'] = judging_perceiving['post'].str.lower()

#print(judging_perceiving[['post']].head())

#Remove URLs
url_pattern = r'http\S+|www\S+|https\S+'

extravert_introvert['post'] = extravert_introvert['post'].str.replace(
	r'http\S+|www\.\S+',
	'',
	regex=True
)
url_pattern = r'http\S+|www\S+|https\S+'


#Normalize punctuation
print("Before punctuation normalization:")
print(extravert_introvert['post'].head())
extravert_introvert['post'] = extravert_introvert['post'].apply(normalize_punctuation)
sensing_intuitive['post'] = sensing_intuitive['post'].apply(normalize_punctuation)
feeling_thinking['post'] = feeling_thinking['post'].apply(normalize_punctuation)
judging_perceiving['post'] = judging_perceiving['post'].apply(normalize_punctuation)

#Remove extra whitespace
extravert_introvert['post'] = extravert_introvert['post'].str.strip()
sensing_intuitive['post'] = sensing_intuitive['post'].str.strip()
feeling_thinking['post'] = feeling_thinking['post'].str.strip()
judging_perceiving['post'] = judging_perceiving['post'].str.strip()

#Tokenization
#I used rule-based tokenizer that separates sentence-final punctuation marks while preserving stylistic markers(exclamation and question marks).
extravert_introvert['post'] = tokenize(extravert_introvert['post'])
sensing_intuitive['post'] = tokenize(sensing_intuitive['post'])
feeling_thinking['post'] = tokenize(feeling_thinking['post'])
judging_perceiving['post'] = tokenize(judging_perceiving['post'])

#Check
print(extravert_introvert['post'].head())
