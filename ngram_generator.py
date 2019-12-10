import numpy as np
from nltk.util import ngrams

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

