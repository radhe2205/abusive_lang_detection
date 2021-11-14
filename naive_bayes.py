import numpy as np
from sklearn.metrics import classification_report
from preprocessing import Preprocessor
import time

class NaiveBayes:
    def __init__(self):
        self.labels = set()
        self.word_counts = {}
        self.priors = {}
        self.likelihoods = {}

    def train(self, train_x, train_y):
        self.labels = set(train_y)
        self.word_counts = {label:{} for label in self.labels}
        self.priors = {label:0 for label in self.labels}
        self.likelihoods = {label:{} for label in self.labels}
        self.total_words = 0

        for tweet,label in zip(train_x,train_y):
            self.priors[label] += 1

            for word in tweet.split():
                self.total_words += 1
                if word in self.word_counts[label]:
                    self.word_counts[label][word] += 1
                else:
                    self.word_counts[label][word] = 1
            
        self.priors = {l:self.priors[l]/len(train_y) for l in self.labels}
        self.likelihoods = {
            l : {
                w : (self.word_counts[l][w]+1)/(self.total_words+sum(self.word_counts[l].values())) \
                    for w in self.word_counts[l]
            } for l in self.labels
        }

    def predict(self, tweet):
        best = {'label':None, 'score':np.inf}

        for l in self.labels:
            score = -np.log(self.priors[l])
            for w in tweet.split():
                if w in self.likelihoods[l]:
                    score += -np.log(self.likelihoods[l][w])
                else:
                    score += -np.log(1.0/(self.total_words+sum(self.word_counts[l].values())))
            
            if score < best['score']:
                best['score'],best['label'] = score,l
            
        return best['label']

    def test(self, test_x):
        return [self.predict(tweet) for tweet in test_x]

    def acc(self, y_preds, y_true):
        return sum([yp==yt for yp,yt in zip(y_preds,y_true)])/len(y_preds)


if __name__ == "__main__":
    print('='*40)
    print('task A')
    print('loading data...')
    pp = Preprocessor()
    train_data = pp.read_tsv(path='data/OLIDv1.0/olid-training-v1.0.tsv')
    test_a_data = pp.read_tsv(path='data/OLIDv1.0/testset-levela.tsv')
    test_a_labels = pp.read_csv(path='data/OLIDv1.0/labels-levela.csv')
    
    print('cleaning data...')
    start = time.time()
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
    end = time.time()
    print(f'took {round(end-start,2)}s') 
    train_x, train_y = pp.get_train_split_by_task(data=train_data, task='subtask_a')
    test_x, test_y = pp.get_test_spilt_by_task(data=test_data, labels=test_a_labels)
    print('='*40)
    print()

    print('='*40)
    print('training model...')
    start = time.time()
    model = NaiveBayes()
    model.train(train_x=train_x, train_y=train_y)
    end = time.time()
    print(f'took {round(end-start,2)}s')
    print('predicting...')
    start = time.time()
    preds = model.test(test_x=test_x)
    end = time.time()
    print(f'took {round(end-start,2)}s')    
    print('='*40)
    print()

    results = classification_report(y_true=test_y, y_pred=preds, output_dict=True, digits=4)
    print('='*40)
    print('testing')
    print(f'accuracy: \t{model.acc(preds,test_y)}')
    print(f'precision: \t{results["macro avg"]["precision"]}')
    print(f'recall: \t{results["macro avg"]["recall"]}')
    print(f'F1-score: \t{results["macro avg"]["f1-score"]}')
    print('='*40)
    print()

# ========================================
# task A
# loading data...
# cleaning data...
# took 0.26s
# ========================================

# ========================================
# training model...
# took 2.8s
# predicting...
# took 0.36s
# ========================================

# ========================================
# testing
# accuracy:       0.7395348837209302
# precision:      0.7725511898173769
# recall:         0.5397177419354838
# F1-score:       0.5019184825888655
# ========================================