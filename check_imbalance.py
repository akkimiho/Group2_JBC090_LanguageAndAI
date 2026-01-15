import pandas as pd
import matplotlib.pyplot as plt

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

#Check imbalances
datasets = {
    "extrovert_introvert": ("data/extrovert_introvert.csv", "extrovert"),
    "sensing_intuitive":   ("data/sensing_intuitive.csv", "sensing"),
    "feeling_thinking":    ("data/feeling_thinking.csv", "feeling"), 
    "judging_perceiving":  ("data/judging_perceiving.csv", "judging"),
}


#Class distribution - searching imbalances
for name, (path, label_col) in datasets.items():
    print(f"\n Processing: {name} => {path}")
    df = pd.read_csv(path)
    print(f"\n{name}")
    print(f"  Unique authors: {df['auhtor_ID'].nunique()}")
    print(f"  Total posts: {len(df)}")
    required_cols = {"auhtor_ID", "post", label_col}
    counts = df[label_col].value_counts()
    perc = df[label_col].value_counts(normalize=True) * 100
    print("Class counts:\n", counts)
    print("Class %:\n", perc.round(2))
    
print('Done')

#Received results from previous code execution
# Class percentages
imbalance_dict = {
    "Extrovert-Introvert": {
        "Extrovert (%)": 22.45,
        "Introvert (%)": 77.55,
    },
    "Sensing-Intuitive": {
        "Sensing (%)": 12.56,
        "Intuitive (%)": 87.44,
    },
    "Feeling-Thinking": {
        "Feeling (%)": 33.92,
        "Thinking (%)": 66.08,
    },
    "Judging-Perceiving": {
        "Judging (%)": 61.56,
        "Perceiving (%)": 38.44,
    }
}

df = pd.DataFrame.from_dict(imbalance_dict, orient="index")

plot_rows = []
for dataset, stats in imbalance_dict.items():
    pct_keys = [k for k in stats.keys() if "(%)" in k]

    k1, k2 = pct_keys
    v1, v2 = float(stats[k1]), float(stats[k2])

    total = v1 + v2
    v1 = v1 / total * 100
    v2 = v2 / total * 100

    plot_rows.append({
        "Dataset": dataset,
        "Trait A": k1.replace(" (%)", ""),
        "A_pct": v1,
        "Trait B": k2.replace(" (%)", ""),
        "B_pct": v2
    })

plot_df = pd.DataFrame(plot_rows)

#Horizontal Stacked bar chart : Class imbalances
fig, ax = plt.subplots(figsize=(10, 5))
y = range(len(plot_df))
ax.barh(y, plot_df["A_pct"])
ax.barh(y, plot_df["B_pct"], left=plot_df["A_pct"])
ax.set_yticks(list(y))
ax.set_yticklabels(plot_df["Dataset"])
ax.set_xlim(0, 100)
ax.set_xlabel("Percentage (%)")
ax.set_title("Class Imbalances in Personality Datasets")

#Add labels
for i, row in plot_df.iterrows():
    a, b = row["A_pct"], row["B_pct"]
    trait_a, trait_b = row["Trait A"], row["Trait B"]

    ax.text(
        a / 2,
        i,
        f"{trait_a}\n{a:.1f}%",
        va="center",
        ha="center",
        fontsize=9,
        color="white"
    )

    ax.text(
        a + b / 2,
        i,
        f"{trait_b}\n{b:.1f}%",
        va="center",
        ha="center",
        fontsize=9,
        color="white" if b > 15 else "black"
    )

plt.tight_layout()
plt.show()