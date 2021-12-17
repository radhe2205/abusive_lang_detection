from preprocessing import Preprocessor
import pandas as pd

def save_cleaned_tweets(data, path):
    pp = Preprocessor()
    cleaned_data = pp.clean_data(data, no_users_url=True,
                   no_html_entities=True,
                   no_hastags=True,
                   all_lowercase=True,
                   no_ascii=True,
                   sep_latin=True,
                   handle_apostrophe=True,
                   no_punc=False,
                   no_numbers=False,
                   no_stop_words=False,
                   reduce_all_words=True,
                   fix_spelling=False,
                   stem_all_words=False)
    cleaned_data.to_csv(path, sep = "\t")

def clean_all_files():
    files = ["data/OLIDv1.0/testset-levela.tsv", "data/OLIDv1.0/testset-levelb.tsv",
             "data/OLIDv1.0/testset-levelc.tsv", "data/OLIDv1.0/olid-training-v1.0.tsv"]

    for file in files:
        data = pd.read_csv(file, sep='\t')
        new_path = ".".join(file.split(".")[:-1]) + "_clean3." + file.split(".")[-1]
        save_cleaned_tweets(data, new_path)
        print(f"Cleaned: {file}")

clean_all_files()
