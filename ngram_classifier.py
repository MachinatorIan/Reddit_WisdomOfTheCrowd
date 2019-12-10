import numpy as np
from collections import Counter, defaultdict
from math import ceil, floor
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.tokenize import word_tokenize

import string

class NgramJudgementClassifier:
    def __init__(self, filenames, N=1, smoothing_constant=0.1, train_prop=0.8, num_stopwords=5):
        self.N = N
        self.n_categories = len(filenames)
        self.stemmer = SnowballStemmer('english')
        self.submissions = []
        for filename in filenames:
            with open(filename) as f:
                splitText = f.read().split('\n')
                category_submissions = []
                for submission in splitText:
                    if len(submission) > 1:
                        try:
                            category_submissions.append(submission.split('\t',1)[1])
                        except:
                            pass
                self.submissions.append(category_submissions)

        balanced_submissions = self.balance_judgements(self.submissions)
        self.train_submissions, self.test_submissions = self.dev_split(balanced_submissions, train_prop=train_prop)
        
        self.train_words = [self.extract_words(submission_set) if self.N == 1 else self.extract_ngrams(submission_set) for submission_set in [submissions for (submissions, lables) in self.train_submissions]]
        self.master_dict = defaultdict()
        self.totals = [0 for _ in range(len(filenames))]
        self.count_and_smooth(self.train_words, smoothing_constant=smoothing_constant)
    
    def balance_judgements(self, submissions, method='undersample'):
        if method == 'undersample':
            balanced_category_len = len(submissions[submissions.index(min(submissions, key=len))])
        if method == 'oversample':
            balanced_category_len = len(submissions[submissions.index(max(submissions, key=len))])

        balanced_submissions = []
        for submission_set in submissions:
            random_order = np.random.choice(a=list(range(len(submission_set))), size=balanced_category_len, replace=False).astype(int)
            random_ordered_submission_set = [submission_set[i] for i in random_order]
            balanced_submissions.append(random_ordered_submission_set)
        return(balanced_submissions)

    def dev_split(self, submissions, train_prop=0.8):
        test = []
        train = []
        for category_ind, category_submissions in enumerate(submissions):
            split_category = train_test_split(category_submissions, [category_ind]*len(category_submissions), train_size=train_prop)
            train.append((split_category[0], split_category[2]))
            test.append((split_category[1], split_category[3]))
        return(train, test)
    
    def tokenize_sentence(self, sent):
        # Split string sentence into tokens using NLTK's word tokenizer.
        try:
            tokenized_sent = word_tokenize(sent)
        except:
            tokenized_sent = []
        
        # Pad each list of tokens with sentence delimiters.
        # '<s>' signifies the start of a sentence.
        # '</s>' signifies the end of a sentence
        tokenized_sent.insert(0, '<s>')
        tokenized_sent.append('</s>')
        return(tokenized_sent)
    
    def tokenize_comment(self, comment_txt):
        # Split comment into sentences using NLTK's sentence detector (trained on engish).
        sentences = sent_detector.tokenize(comment_txt.strip())

        # Tokenize each sentence and enclose in sentence delimiters.
        tokenized_sentences = [self.tokenize_sentence(sent) for sent in sentences]

        # Enclose the comment in comment delimiters.
        # Start each comment with a '<c>'
        tokenized_comment = ['<c>']
        # Add tokenized sentences.
        for sent in tokenized_sentences:
            tokenized_comment += sent
        # End each comment with a '</c>'
        tokenized_comment += ['</c>']

        return(tokenized_comment)

    def extract_words(self, submissions):
        words = []
        for submission in submissions:
            for word in submission.split(' '):
                words.append(self.stemmer.stem(word.lower()))
        return(words)
    
    def extract_ngrams(self, submissions):
        n_grams = []
        for submission in submissions:
            tokenized_submission = self.tokenize_comment(submission)
            stemmed_submission = list(map(lambda token: self.stemmer.stem(token.lower()), tokenized_submission))
            n_grams += list(ngrams(stemmed_submission, self.N))
        return(n_grams)

    def count_and_smooth(self, words_by_judgement, smoothing_constant=0.1, count_thresh=1):
        counter_dicts = [Counter(words) for words in words_by_judgement]

        words_all_categories = set(counter_dicts[0].keys())
        for counter_dict in counter_dicts[1:]:
            words_all_categories = words_all_categories.union(set(counter_dict.keys())) 
        
        for word in words_all_categories:
            word_count_by_category = [counter_dict[word] for counter_dict in counter_dicts]
            
            for i, word_count in enumerate(word_count_by_category):
                freqZeroCategories = list(range(len(word_count_by_category)))
                if not(word_count):
                    word_count_by_category[i] = smoothing_constant
                    freqZeroCategories.remove(i)
            
            for i in freqZeroCategories:
                word_count_by_category[i] += smoothing_constant

            if any([1 if word_count > count_thresh else 0 for word_count in word_count_by_category]):
                self.master_dict[word] = word_count_by_category
                self.totals = [sum([counts[j] for counts in self.master_dict.values()]) for j in range(self.n_categories)]
            
    def uni_prob(self, word):
        try:
            counts = self.master_dict[word]
        except: 
            return(0, 0)

        probs = [self.master_dict[word][i]/self.totals[i] for i in range(self.n_categories)]
        return(probs)
    
    def predict(self, review):
        review = review.translate(str.maketrans('', '', string.punctuation))
        review = self.extract_words([review])
        
        logProbs = [0 for _ in range(self.n_categories)]
        
        for word in review:
            probs = self.uni_prob(word)
            if(any(probs)):
                for i in range(len(logProbs)):
                    logProbs[i] += np.log(probs[i])
        
        max_prob = max(logProbs)
        max_category = logProbs.index(max_prob)
        return(max_category)
        #return(list(range(self.n_categories)).index(max(list(range(self.n_categories)))))
    
    def run_test(self, test_submissions):
        correct = 0
        labeled_submission_tups = []
        for (category_submissions, category_labels) in test_submissions:
            labeled_submission_tups += list(zip(category_submissions, category_labels))
        for submission, label in labeled_submission_tups:
            pred = self.predict(submission)
            if(pred == label):
                correct += 1
                
        return(correct/len(labeled_submission_tups))
    
    def run_splitTest(self):
        result = self.run_test(self.test_submissions)
        print(result)
