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

print(judging_perceiving[['post']].head())


