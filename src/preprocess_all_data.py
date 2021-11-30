from preprocessing import Preprocessor
import pandas as pd

def save_cleaned_tweets(data, path):
    pp = Preprocessor()
    cleaned_data = pp.clean_data(data)
    cleaned_data.to_csv(path, sep = "\t")

def clean_all_files():
    files = ["data/OLIDv1.0/testset-levela.tsv", "data/OLIDv1.0/testset-levelb.tsv",
             "data/OLIDv1.0/testset-levelc.tsv", "data/OLIDv1.0/olid-training-v1.0.tsv"]

    for file in files:
        data = pd.read_csv(file, sep='\t')
        new_path = ".".join(file.split(".")[:-1]) + "_clean1." + file.split(".")[-1]
        save_cleaned_tweets(data, new_path)
        print(f"Cleaned: {file}")

clean_all_files()
