import time
import pandas as pd	
from data_preprocessing import extrovert_introvert, sensing_intuitive, feeling_thinking, judging_perceiving

#Check imbalances
print(extrovert_introvert['label'].value_counts())
print(sensing_intuitive['label'].value_counts())
print(feeling_thinking['label'].value_counts())
print(judging_perceiving['label'].value_counts())	
