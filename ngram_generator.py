import numpy as np
from nltk.util import ngrams

from collections import Counter, defaultdict
from math import ceil, floor
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
#import numpy as np
import string

class NgramCommentGenerator():
    def __init__(self, corpus, N):
        self.corpus = corpus
        self.N = N
        self.flat_corpus = self.flatten_corpus()
        self.ngram_corpus = {}
        for judgement_category in self.flat_corpus.keys():
            self.ngram_corpus[judgement_category] = self.make_ngram_corpus(self.flat_corpus[judgement_category])
   
    def make_ngram_corpus(self, corpus):
        ngram_tups_list = list(ngrams(corpus, self.N))
        ngram_corpus = []
        for gi in ngram_tups_list:
            if not gi[0:self.N-1] in [x[0] for x in ngram_corpus]:
                next_word_dict = {}
                
                for gj in ngram_tups_list:
                    if gj[0:self.N-1] == gi[0:self.N-1]:
                        if gj[self.N-1] in next_word_dict.keys():
                            next_word_dict[gj[self.N-1]] += 1
                        else:
                            next_word_dict[gj[self.N-1]] = 1
                gi_count = sum(next_word_dict.values())
                next_word_prob_tups = tuple([(key, value/gi_count) for key, value in next_word_dict.items()])
                ngram_corpus.append((gi[0:self.N-1], gi_count, next_word_prob_tups))
        return(ngram_corpus)

    def flatten_corpus(self):
        flat_comment_corpus = dict(zip(self.corpus.keys(), [[] for _ in range(len(self.corpus.keys()))]))
        for judgement_category in self.corpus.keys():
            for (_, comment) in self.corpus[judgement_category]:
                for token in comment:
                    flat_comment_corpus[judgement_category].append(token)
        return(flat_comment_corpus)
    
    def generate_comment(self, judgement_category):
        corpus = self.ngram_corpus[judgement_category]
        comment_starts = []
        for gi  in corpus:
            if gi[0][0] == '<c>':
                comment_starts.append(gi)
        num_starts = sum([gi[1] for gi in comment_starts])
        probs=[gi[1]/num_starts for gi in comment_starts]
        comment_start = comment_starts[np.random.choice(list(range(len(comment_starts))), p=probs)]
        comment = [w for w in comment_start[0]]
        while comment[-1] != '</c>':
            prev_gram = tuple(comment[-(self.N-1):])
            next_choices = None
            for gi in corpus:
                if gi[0] == prev_gram:
                    next_choices = gi[2]
            if next_choices is not None:
                next_probs = [x[1] for x in next_choices]
                next_word = next_choices[np.random.choice(list(range(len(next_choices))),p=next_probs)][0]
                comment.append(next_word)
            else:
                print('N-Gram not found in corpus')
        return(' '.join(comment))

class NgramSubmissionClassifier():
    def __init__(self, corpus, N):
        self.corpus = corpus
        self.N = N
        self.flat_corpus = self.flatten_corpus()
        self.ngram_corpus = {}
        for judgement_category in self.flat_corpus.keys():
            self.ngram_corpus[judgement_category] = self.make_ngram_corpus(self.flat_corpus[judgement_category])
   
    def make_ngram_corpus(self, corpus):
        ngram_tups_list = list(ngrams(corpus, self.N))
        ngram_corpus = []
        for gi in ngram_tups_list:
            if not gi[0:self.N-1] in [x[0] for x in ngram_corpus]:
                next_word_dict = {}
                
                for gj in ngram_tups_list:
                    if gj[0:self.N-1] == gi[0:self.N-1]:
                        if gj[self.N-1] in next_word_dict.keys():
                            next_word_dict[gj[self.N-1]] += 1
                        else:
                            next_word_dict[gj[self.N-1]] = 1
                gi_count = sum(next_word_dict.values())
                next_word_prob_tups = tuple([(key, value/gi_count) for key, value in next_word_dict.items()])
                ngram_corpus.append((gi[0:self.N-1], gi_count, next_word_prob_tups))
        return(ngram_corpus)

    def flatten_corpus(self):
        flat_comment_corpus = dict(zip(self.corpus.keys(), [[] for _ in range(len(self.corpus.keys()))]))
        for judgement_category in self.corpus.keys():
            for (_, comment) in self.corpus[judgement_category]:
                for token in comment:
                    flat_comment_corpus[judgement_category].append(token)
        return(flat_comment_corpus)
    
    def generate_comment(self, judgement_category):
        corpus = self.ngram_corpus[judgement_category]
        comment_starts = []
        for gi  in corpus:
            if gi[0][0] == '<c>':
                comment_starts.append(gi)
        num_starts = sum([gi[1] for gi in comment_starts])
        probs=[gi[1]/num_starts for gi in comment_starts]
        comment_start = comment_starts[np.random.choice(list(range(len(comment_starts))), p=probs)]
        comment = [w for w in comment_start[0]]
        while comment[-1] != '</c>':
            prev_gram = tuple(comment[-(self.N-1):])
            nnext_choices = None
            for gi in corpus:
                if gi[0] == prev_gram:
                    next_choices = gi[2]
            if next_choices is not None:
                next_probs = [x[1] for x in next_choices]
                next_word = next_choices[np.random.choice(list(range(len(next_choices))),p=next_probs)][0]
                comment.append(next_word)
            else:
                print('N-Gram not found in corpus')
        return(' '.join(comment))

# Class ReviewClassifier will be used to create a Naive Bayes classifier object that is in itialized with
# the filenames of files containing positive and negative examples, and a default 80/20 split of training to 
# test data.The train_prop argument can be used to specify different split.
class ReviewClassifier:
    def __init__(self, filenames, train_prop=0.8, num_stopwords=5):
        self.stemmer = SnowballStemmer('english')
        
        splitTexts = [open(filename).read().split('\n') for filename in filenames]
        #pos_splitText = open(pos_filename).read().split('\n')
        #neg_splitText = open(neg_filename).read().split('\n')
        
        submissions = []
        for splitText in splitTexts:
            try:
                submissions.append([review.split('\t')[1] for review in splitText if len(review) > 1])
            except:
                pass
        #pos_reviews = [review.split('\t')[1] for review in pos_splitText if len(review) > 1]
        #neg_reviews = [review.split('\t')[1] for review in neg_splitText if len(review) > 1]
        
        #pos, neg = self.dev_split(pos_reviews, neg_reviews, train_prop=train_prop)
        
        self.train_words = [self.extract_words(sumbission_set) for sumbission_set in submissions]
        #self.train_pos_words = self.extract_words(pos[0])
        #self.train_neg_words = self.extract_words(neg[0])
        
        #self.test_reviews = [(review, 1) for review in pos[1]] + [(review, 0) for review in neg[1]]
        
        self.master_dict = defaultdict()
        self.totals = [0 for _ in range(len(filenames))]
        #self.pos_total = 0
        #self.neg_total = 0
        self.n_categories = len(filenames)
        self.count_and_smooth(self.train_words)
    
    def dev_split(self, pos, neg, train_prop=0.8):
        split_pos = train_test_split(pos, [1]*len(pos), train_size=train_prop)
        split_neg = train_test_split(neg, [0]*len(neg), train_size=train_prop)
        return(split_pos, split_neg)
    
    def extract_words(self, reviews):
        words = []
        for review in reviews:
            for word in review.split(' '):
                words.append(self.stemmer.stem(word.lower()))
        return(words)
    
    def count_and_smooth(self, words_by_judgement, smoothing_constant=0.5, count_thresh=1):
        
        counter_dicts = [Counter(words) for words in words_by_judgement]
        #pos_dict = Counter(pos_words)
        #neg_dict = Counter(neg_words)
        
        words_all_categories = set(counter_dicts[0].keys())
        for counter_dict in counter_dicts[1:]:
            words_all_categories = words_all_categories.union(set(counter_dict.keys())) 
        #words = set(pos_dict.keys()).union(set(neg_dict.keys()))
        
        for word in words_all_categories:
            word_count_by_category = [counter_dict[word] for counter_dict in counter_dicts]
            #neg_word_count = neg_dict[word]
            #pos_word_count = pos_dict[word]
            
            for i, word_count in enumerate(word_count_by_category):
                freqZeroCategories = list(range(len(word_count_by_category)))
                if not(word_count):
                    word_count_by_category[i] = smoothing_constant
                    freqZeroCategories.remove(i)
            
            for i in freqZeroCategories:
                word_count_by_category[i] += smoothing_constant

            if any([1 if word_count > count_thresh else 0 for word_count in word_count_by_category]):
                self.master_dict[word] = word_count_by_category
                for i in range(len(self.totals)):
                    self.totals[i] = sum([i[j] for j in range(len(word_count_by_category)) for i in master_dict.values])
                #self.pos_total = sum([i[0] for i in self.master_dict.values()])
                #self.neg_total = sum([i[1] for i in self.master_dict.values()])
            
    def uni_prob(self, word):
        try:
            counts = self.master_dict[word]
        except: 
            return(0, 0)
        
        #print(counts)
        probs = [self.master_dict[word][i]/self.totals[i] for i in range(len(master_dict[word]))]
        #pos_prob = self.master_dict[word][0]/self.pos_total
        #neg_prob = self.master_dict[word][1]/self.neg_total
        #return(pos_prob, neg_prob)
        return(probs)
    
    def predict(self, review):
        review = review.translate(str.maketrans('', '', string.punctuation))
        review = self.extract_words([review])
        
        logProbs = [0 for _ in range(self.n_categories)]
        #pos_logProb = 0
        #neg_logProb = 0
        
        for word in review:
            probs = self.uni_prob(word)
            #pos_word_prob, neg_word_prob = self.uni_prob(word)
            
            if(any(probs)):
                for i in range(len(logProbs)):
                    logProbs[i] += np.log(probs[i])
                #pos_logProb += np.log(pos_word_prob) 
                #neg_logProb += np.log(neg_word_prob)
            
        
        #if(pos_logProb > neg_logProb):
        #    return(1)
        #else:
        #    return(0)
        return(list(range(self.n_categories)).index(max(list(range(self.n_categories)))))
    
    def run_test(self, test_reviews):
        correct = 0
        for review in test_reviews:
            pred = self.predict(review[0])
            if(pred == review[1]):
                correct += 1
                
        return(correct/len(test_reviews))
    
    def run_splitTest(self):
        result = self.run_test(self.test_reviews)
        print(result)
