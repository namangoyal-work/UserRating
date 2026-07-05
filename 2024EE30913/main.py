import numpy as np
import pandas as pd
import nltk
nltk.download('popular', quiet=True)
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re, sys, pickle

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

from xgboost import XGBClassifier

def dummy(doc):
    return doc

def create_feature(l, f):
    return [f(e) for e in l]

class SentimentClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, C=1.0, dim=2000, max_iter=100):
        self.tfidf = TfidfVectorizer(analyzer='word', tokenizer=dummy,
            preprocessor=dummy, token_pattern=None)
        self.tfidf_bigram = TfidfVectorizer(analyzer='word', tokenizer=dummy,
            preprocessor=dummy, token_pattern=None)

        self.svd = TruncatedSVD(n_components=dim)
        self.svd_bigram = TruncatedSVD(n_components=dim)
        
        self.lr     = LogisticRegression(solver='lbfgs', C=1, max_iter=100)
        self.lr_bal = LogisticRegression(solver='lbfgs', class_weight='balanced', C=1, max_iter=100)
        self.lr_bg  = LogisticRegression(solver='lbfgs', C=1, max_iter=100)
        
        self.xgb = XGBClassifier(
            max_depth=6,
            colsample_bytree=0.2,
            colsample_bynode=0.8,
            n_estimators=20,
            objective='multi:softmax',
            learning_rate=0.3
        )
        
    def transform(self, X):
        
        # tokenize
        X = create_feature(X, word_tokenize)

        # lowercase
        X = create_feature(X, lambda x: [w.lower() for w in x])

        # PoS tagging
        X = create_feature(X, nltk.pos_tag)
        
        # negation via looking at PoS: negate all adjectives/verbs from a not/n't to the next 
        # stopword/punctuation mark. Also do lower case
        def negate(pos_arr):
            neg_pos_arr = []
            negating = False
            for word in pos_arr:
                w = word[0]
                if (w == 'not' or w == "n't"):
                    negating = True
                    neg_pos_arr.append(("POS", w, word[1]))
                    continue
                elif (word[1] == '.' or word[1] == ':' or word[1] == 'IN' or word[1] == 'CC'):
                    negating = False
                    neg_pos_arr.append(("POS", w, word[1]))
                    continue

                if negating:
                    neg_pos_arr.append(("NEG", w, word[1]))
                else:
                    neg_pos_arr.append(("POS", w, word[1]))

            return neg_pos_arr

        X = create_feature(X, negate)
        
        # stopword and punctuation removal
        sw = set(stopwords.words('english'))

        def stopword_removal(pos_arr):
            char_regex = re.compile(r"[A-Z]+")
            word_arr = []
            for word in pos_arr:
                if (word[1] not in sw):
                    word_arr.append(word)
            return word_arr

        X = create_feature(X, stopword_removal)
        
        # lemmatization
        wnl = WordNetLemmatizer()

        def lemmatize(pos_arr):
            # POS tag conversion
            pos_tag_map = {
                'J': 'a', # adjective
                'N': 'n', # noun
                'V': 'v', # verb
                'R': 'r'  # adverb
            }
            lemmatized_list = []
            for word in pos_arr:
                if word[2][0] in pos_tag_map:
                    lemmatized_list.append((word[0], wnl.lemmatize(word[1], pos=pos_tag_map[word[2][0]]), word[2]))
                else:
                    lemmatized_list.append(word)
            return lemmatized_list

        X = create_feature(X, lemmatize)
        
        return X
    
    def fit_ll_scores(self, X, y):
        # Zipf law encoding and dropping extraneous words
        tok_set = set()
        for review in X:
            for tok in review:
                tok_set.add(tok)

        toks = list(tok_set)
        rev_toks = {tok:i for i,tok in enumerate(toks)}
        tok_freq = np.zeros((5,len(toks)))

        for i, review in zip(y, X):
            for tok in review:
                tok_freq[i-1,rev_toks[tok]] += 1
        
        tok_freq_lists = [sorted([(f,i) for i,f in enumerate(tok_freq[c])])[::-1][:1000] for c in range(5)]
        tok_freqs = set()
        for tfl in tok_freq_lists:
            tok_freqs = tok_freqs.union(set([i for f,i in tfl]))

        tok_freq_w = tok_freq.sum(axis=0)
        tok_freq_c = tok_freq.sum(axis=1)
        tot_tok = tok_freq.sum()

        def P_w(w):
            return (tok_freq_w[w]+5)/(tot_tok+5*len(toks))

        def P_w_c(w, c):
            return (tok_freq[c,w]+1)/(tok_freq_c[c]+len(toks))

        tok_lls = [([P_w_c(i,c)/P_w(i) for c in range(5)],i) for i in tok_freqs]
        tok_ll_ratios = [(max(l)/min(l),i) for l,i in tok_lls]
        top_ratios = sorted(tok_ll_ratios)[::-1]

        top_ratios_shortlisted = top_ratios[:200]

        self.bigram_toks = set(toks[i] for r,i in top_ratios_shortlisted)

        # feature goodness calc
        self.tok_ll_dict = {toks[b]: np.log(np.array(a)) for a,b in tok_lls}
        
    def predict_ll_scores(self, X):
        ll_scores = []
        for pos_tokens in X:
            tok_lls = np.zeros(5)
            for tok in pos_tokens:
                if tok in self.tok_ll_dict:
                    tok_lls += self.tok_ll_dict[tok]
            ll_scores.append(tok_lls)

        return np.array(ll_scores)
    
    def extract_bigrams(self, pos_arr):
        bigrams = []
        n = len(pos_arr)
        tok_arr = [r[1] for r in pos_arr]

        for i in range(n-1):
            if pos_arr[i] in self.bigram_toks or \
               (i == n-2 and pos_arr[i+1] in self.bigram_toks):
                bigrams.append("-".join(tok_arr[i:i+2]))

        return bigrams

    def fit(self, X, y):
        
        X = self.transform(X)
        
        self.fit_ll_scores(X, y)
        
        bigrams = create_feature(X, self.extract_bigrams)
        
        X_str = [[f"{t[0]}_{t[1]}_{t[2]}" for t in x] for x in X]
        
        train_tfidf = self.tfidf.fit_transform(X_str)
        train_svd = self.svd.fit_transform(train_tfidf)

        train_bigram_tfidf = self.tfidf_bigram.fit_transform(bigrams)
        train_bigram_svd = self.svd_bigram.fit_transform(train_bigram_tfidf)

        self.train_svd_mean = train_svd.mean(axis=0)
        self.train_svd_std = train_svd.std(axis=0)
        train_svd_white = (train_svd-self.train_svd_mean)/self.train_svd_std
        
        self.train_bigram_svd_mean = train_bigram_svd.mean(axis=0)
        self.train_bigram_svd_std = train_bigram_svd.std(axis=0)
        train_bigram_svd_white = (train_bigram_svd-self.train_bigram_svd_mean)/self.train_bigram_svd_std
        
        self.lr.fit(train_svd_white, y)
        self.lr_bal.fit(train_svd_white, y)
        self.lr_bg.fit(train_bigram_svd_white, y)
        
        train_log_probs = np.hstack([
            self.lr.predict_log_proba(train_svd_white), 
            self.lr_bal.predict_log_proba(train_svd_white), 
            self.lr_bg.predict_log_proba(train_bigram_svd_white), 
            self.predict_ll_scores(X)
        ])
        
        self.xgb.fit(train_log_probs, np.array(y)-1)

    def predict(self, X):
        
        X = self.transform(X)
        
        bigrams = create_feature(X, self.extract_bigrams)
        
        X_str = [[f"{t[0]}_{t[1]}_{t[2]}" for t in x] for x in X]
        
        X_tfidf = self.tfidf.transform(X_str)
        X_svd = self.svd.transform(X_tfidf)

        X_bigram_tfidf = self.tfidf_bigram.transform(bigrams)
        X_bigram_svd = self.svd_bigram.transform(X_bigram_tfidf)

        X_svd_white = (X_svd-self.train_svd_mean)/self.train_svd_std
        X_bigram_svd_white = (X_bigram_svd-self.train_bigram_svd_mean)/self.train_bigram_svd_std
        
        X_log_probs = np.hstack([
            self.lr.predict_log_proba(X_svd_white), 
            self.lr_bal.predict_log_proba(X_svd_white), 
            self.lr_bg.predict_log_proba(X_bigram_svd_white), 
            self.predict_ll_scores(X)
        ])
        
        return self.xgb.predict(X_log_probs)+1

def eval_metrics(true, preds):
    f1_micro = f1_score(true, preds, average='micro') 
    f1_macro = f1_score(true, preds, average='macro')
    print(f"    F1 micro: {f1_micro}")
    print(f"    F1 macro: {f1_macro}")
    print(f"    Final score: {(f1_micro+f1_macro)/2}")
    return f1_micro, f1_macro
    
if __name__ == '__main__':
    
    # DEBUGGING
    # sys.argv = ['main.py', 'train', '/kaggle/input/col772-a1-data/train.csv', '/kaggle/working/trained_model']
    # sys.argv = ['main.py', 'test', '/kaggle/working/trained_model', '/kaggle/input/col772-a1-data/sample_input.csv', '/kaggle/working/sample_output.csv']
    # sys.argv = ['main.py', 'cv', '/kaggle/input/col772-a1-data/train.csv']
    
    if sys.argv[1] == 'train':
        # train
        model = SentimentClassifier()
        train = pd.read_csv(sys.argv[2], header=None).dropna().iloc[:1000]
        model.fit(list(train[0]), list(train[1]))
        pickle.dump(model, open(sys.argv[3], 'wb'))
        
    elif sys.argv[1] == 'test':
        # predict
        model_path = sys.argv[2]
        test = pd.read_csv(sys.argv[3], header=None)
        outpath = sys.argv[4]
        
        model = pickle.load(open(model_path, 'rb'))
        preds = model.predict(test[0])
        np.savetxt(outpath, preds, fmt='%d')
        
    elif sys.argv[1] == 'cv':
        # cross validate
        model = SentimentClassifier()
        df = pd.read_csv(sys.argv[2], header=None).dropna().iloc[:1000]
        data, labels = list(df[0]), df[1].to_numpy()
        skf = StratifiedKFold(n_splits=5)
        f1m = []
        f1M = []
        for i, (train, val) in enumerate(skf.split(df, df[1])):
            print(f'Fold {i}:')
            model.fit(list(df.iloc[train][0]), df.iloc[train][1])
            preds = model.predict(list(df.iloc[val][0]))
            f1micro, f1macro = eval_metrics(df.iloc[val][1], preds)
            f1m.append(f1micro)
            f1M.append(f1macro)
        
        print()
        print('Averaged metrics:')
        print(f'    F1 micro: {sum(f1m)/5}')
        print(f'    F1 macro: {sum(f1M)/5}')
        print(f'    Final Score: {(sum(f1m)+sum(f1M))/10}')
        
    else:
        print('Unrecognized option.')
