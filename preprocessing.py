import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class Preprocessor:
    def __init__(self):
        pass
    
    def read_tsv(self, path):
        return pd.read_csv(path, sep='\t')
    
    def read_csv(self, path):
        return pd.read_csv(path, names=['id','label'])
    
    def clean_data(self,
                   data,
                   lower_case,
                   remove_hastag,
                   remove_user,
                   remove_url,
                   remove_punc,
                   remove_non_alpha,
                   remove_stop):

        def clean(text):
            if remove_url:
                text = re.sub('URL', '', text)
            if lower_case:
                text = text.lower()
            if remove_hastag:
                text = re.sub('#[A-Za-z0-9_]+', '', text)
            if remove_user:
                text = re.sub('@[A-Za-z0-9_]+', '', text)
            if remove_punc:
                text = re.sub('[()!?]', ' ', text)
                text = re.sub('\[.*?\]', ' ', text)
            if remove_non_alpha:
                text = re.sub('[^a-z0-9]', ' ', text)
            if remove_stop:
                text = text.split()
                stop = set(nltk.corpus.stopwords.words('english'))
                text = [w for w in text if w not in stop]
                text = ' '.join(text)

            text = text.split()
            text = [w for w in text if w != '']
            text = ' '.join(text)
            return text

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

    def get_train_split_by_task(self, data, task, val_set=False):
        labels = data[task]
        tweets = data[~labels.isnull()]['tweet']
        labels = labels[~labels.isnull()]

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


if __name__ == "__main__":
    # example of how to use preprocessor
    pp = Preprocessor()
    train_data = pp.read_tsv(path='data/OLIDv1.0/olid-training-v1.0.tsv')
    test_a_data = pp.read_tsv(path='data/OLIDv1.0/testset-levela.tsv')
    test_a_labels = pp.read_csv(path='data/OLIDv1.0/labels-levela.csv')
    
    train_data = pp.clean_data(data=train_data,
                               lower_case=True, 
                               remove_hastag=True, 
                               remove_user=True, 
                               remove_url=True, 
                               remove_punc=True, 
                               remove_non_alpha=True, 
                               remove_stop=False)
    
    test_data = pp.clean_data(data=test_a_data,
                              lower_case=True, 
                              remove_hastag=True, 
                              remove_user=True, 
                              remove_url=True, 
                              remove_punc=True, 
                              remove_non_alpha=True, 
                              remove_stop=False)
    
    train_x, train_y = pp.get_train_split_by_task(data=train_data, task='subtask_a')
    test_x, test_y = pp.get_test_spilt_by_task(data=test_data, labels=test_a_labels)

    vocab = pp.gen_vocab(sentences=train_x)
    word_to_idx = pp.gen_word_to_idx(vocab=vocab)