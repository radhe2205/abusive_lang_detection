# https://catriscode.com/2021/05/01/tweets-cleaning-with-python/
# https://stackoverflow.com/questions/40550349/insert-spaces-before-and-after-special-symbols-in-python
# https://stackoverflow.com/questions/43018030/replace-apostrophe-short-words-in-python
# https://stackoverflow.com/questions/817122/delete-digits-in-python-regex
# https://github.com/mammothb/symspellpy
# https://www.geeksforgeeks.org/python-lemmatization-with-nltk/

import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import string
from nltk.stem import WordNetLemmatizer
import pkg_resources
from symspellpy.symspellpy import SymSpell

class Preprocessor:
    def __init__(self):
        pass

    def get_train_data(self, data_path, task = "subtask_a"):
        train_data = self.read_tsv(path=data_path)

        train_data = self.clean_data(data=train_data,no_users_url=False,
                   no_html_entities=False,
                   no_hastags=False,
                   all_lowercase=False,
                   no_ascii=False,
                   sep_latin=False,
                   handle_apostrophe=False,
                   no_punc=False,
                   no_numbers=False,
                   no_stop_words=False,
                   reduce_all_words=False,
                   fix_spelling=False,
                   stem_all_words=False)

        return self.get_train_data_n_labels(train_data, task)

    def get_test_data(self, data_path, label_path):
        test_data = self.read_tsv(path=data_path)
        test_labels = self.read_csv(path=label_path)
        test_data = self.clean_data(data=test_data,no_users_url=False,
                   no_html_entities=False,
                   no_hastags=False,
                   all_lowercase=False,
                   no_ascii=False,
                   sep_latin=False,
                   handle_apostrophe=False,
                   no_punc=False,
                   no_numbers=False,
                   no_stop_words=False,
                   reduce_all_words=False,
                   fix_spelling=False,
                   stem_all_words=False)
        return test_data["tweet"].values, test_labels["label"].values

    def read_tsv(self, path):
        return pd.read_csv(path, sep='\t')
    
    def read_csv(self, path):
        return pd.read_csv(path, names=['id','label'])

    def remove_mention_url(self, text):
        text = re.sub('@[A-Za-z0-9_]+', '', text)
        text = re.sub('URL', '', text)
        return text

    def remove_entities(self, text):
        text = text.replace('&lt;', '')
        text = text.replace('&gt;', '')
        text = text.replace('&amp;', '')
        return text

    def remove_hastags(self, text):
        text = re.sub('#[A-Za-z0-9_]+', '', text)
        return text

    def lowercase(self, text):
        text = text.lower()
        return text

    def remove_non_ascii(self, text):
        text = text.encode('ascii', 'ignore')
        return str(text)

    def add_space_latin(self, text):
        text = re.sub('([.()!"#$%&*+,-/:;<=>?@^_`{|}~])', '\\1', text)
        return text

    def apostrophe_handling(self, text):
        contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i had",
            "i'd've": "i would have",
            "i'll": "i will",
            "i'll've": "i will have",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she had",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we had",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you had",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have"
        }

        for word in text.split():
            if word in contractions:
                text = text.replace(word, contractions[word])
        return text

    def remove_punc(self, text):
        text = re.sub('[()!?.,:;&@#*$%^+=-]', ' ', text)
        text = re.sub('\[.*?\]', ' ', text)
        return text

    def remove_numbers(self, text):
        text = re.sub("\d+", '', text)
        return text

    def remove_stop(self, text):
        text = text.split()
        stop = set(nltk.corpus.stopwords.words('english'))
        text = [w for w in text if w not in stop]
        text = ' '.join(text)
        return text

    def reduce_words(self, text):
        def reduced_word(w):
            s = w[0]
            curr_char = w[0]
            curr_count = 1
            for c in w[1:]:
                if c == curr_char:
                    curr_count += 1
                else:
                    curr_char = c
                    curr_count = 1

                if curr_count <= 2:
                    s += c
                else:
                    continue
            return s

        text = reduced_word(w=text)
        return text    

    def spell_correction(self, text, sym_spell):
        text = sym_spell.word_segmentation(text)
        return text.corrected_string

    def stem_words(self, text, lemmatizer):
        for word in text.split():
            text = text.replace(word, lemmatizer.lemmatize(word))
        return text

    def clean_data(self,
                   data,
                   no_users_url=True,
                   no_html_entities=True,
                   no_hastags=True,
                   all_lowercase=True,
                   no_ascii=True,
                   sep_latin=True,
                   handle_apostrophe=True,
                   no_punc=True,
                   no_numbers=True,
                   no_stop_words=True,
                   reduce_all_words=True,
                   fix_spelling=True,
                   stem_all_words=True):

        def clean(text):
            if no_users_url:
                text = self.remove_mention_url(text=text)
            if no_html_entities:
                text = self.remove_entities(text=text)
            if no_hastags:
                text = self.remove_hastags(text=text)
            if all_lowercase:
                text = self.lowercase(text=text)
            if no_ascii:
                text = self.remove_non_ascii(text=text)
            if sep_latin:
                text = self.add_space_latin(text=text)
            if handle_apostrophe:
                text = self.apostrophe_handling(text=text)
            if no_punc:
                text = self.remove_punc(text=text)
            if no_numbers:
                text = self.remove_numbers(text=text)
            if no_stop_words:
                text = self.remove_stop(text=text)
            if reduce_all_words:
                text = self.reduce_words(text=text)
            if fix_spelling:
                text = self.spell_correction(text=text, sym_spell=sym_spell)
            if stem_all_words:
                text = self.stem_words(text=text, lemmatizer=lemmatizer)

            text = text.split()
            text = [w for w in text if w != '']
            text = ' '.join(text)
            return text

        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

        lemmatizer = WordNetLemmatizer()

        data['tweet'] = data['tweet'].apply(clean)
        return data

    def gen_vocab(self, sentences):
        vocab = set()
        for sentence in sentences:
            words = sentence.split()
            vocab |= set(words)
        return vocab

    def gen_word_to_idx(self, vocab):
        n = len(vocab)
        word_to_idx = {word:i for i,word in enumerate(vocab)}
        word_to_idx['<UNK>'] = n
        return word_to_idx

    def get_train_data_n_labels(self, data, task):
        labels = data[task]
        tweets = data[~labels.isnull()]['tweet']
        labels = labels[~labels.isnull()]
        return tweets, labels

    def get_train_split_by_task(self, data, task, val_set=False):
        tweets, labels = self.get_train_data_n_labels(data, task)

        if val_set:
            train_x, val_x, train_y, val_y = train_test_split(tweets.values,
                                                              labels.values,
                                                              test_size=0.1,
                                                              stratify=labels,
                                                              random_state=0)
            return train_x, val_x, train_y, val_y
        
        return tweets.values, labels.values

    def get_test_spilt_by_task(self, data, labels):
        return data['tweet'].values, labels['label'].values

def main():
    # example of how to use preprocessor
    pp = Preprocessor()
    train_data = pp.read_tsv(path='data/OLIDv1.0/olid-training-v1.0.tsv')
    test_a_data = pp.read_tsv(path='data/OLIDv1.0/testset-levela.tsv')
    test_a_labels = pp.read_csv(path='data/OLIDv1.0/labels-levela.csv')
    
    train_data = pp.clean_data(data=train_data)
    test_data = pp.clean_data(data=test_a_data)
    
    train_x, train_y = pp.get_train_split_by_task(data=train_data, task='subtask_a')
    test_x, test_y = pp.get_test_spilt_by_task(data=test_data, labels=test_a_labels)
    print(train_x)
    vocab = pp.gen_vocab(sentences=train_x)
    word_to_idx = pp.gen_word_to_idx(vocab=vocab)

def text_clean_unit_tests():

    s = "@USER thissssssss is a test, URL That NNEEEEEEEEDS!!!! to be properly cleand. me+you=forever! &gt;&gt;&gt; THE8 worst 99 &lt;&lt; &amp; :) #loSER abc... i'm didn't know you'd be this's compliccated &*(#$)@#$"
    pp = Preprocessor()
    print(f'raw string: {s}')
    print('remove_mention_url')
    print(pp.remove_mention_url(s))
    print('remove_entities')
    print(pp.remove_entities(s))
    print('remove_hastags')
    print(pp.remove_hastags(s))
    print('lowercase')
    print(pp.lowercase(s))
    print('add_space_latin')
    print(pp.add_space_latin(s))
    print('apostrophe_handling')
    print(pp.apostrophe_handling(pp.lowercase(s)))
    print('remove_punc')
    print(pp.remove_punc(s))
    print('remove_numbers')
    print(pp.remove_numbers(s))
    print('remove_stop')
    print(pp.remove_stop(s))
    print('reduce_words')
    print(pp.reduce_words(s))
    
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    lemmatizer = WordNetLemmatizer()

    print('spell_correction')
    print(pp.spell_correction(s,sym_spell))
    print('stem_words')
    print(pp.stem_words(pp.lowercase(s),lemmatizer))
    print('\nall together')
    s = pp.remove_mention_url(s)
    s = pp.remove_entities(s)
    s = pp.remove_hastags(s)
    s = pp.lowercase(s)
    s = pp.remove_non_ascii(s)
    s = pp.apostrophe_handling(s)
    s = pp.remove_punc(s)
    s = pp.remove_numbers(s)
    s = pp.remove_stop(s)
    s = pp.reduce_words(s)
    s = pp.spell_correction(s,sym_spell)
    s = pp.stem_words(s,lemmatizer)
    print(s)

if __name__ == "__main__":
    main()