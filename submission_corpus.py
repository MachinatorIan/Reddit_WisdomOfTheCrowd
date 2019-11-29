from pprint import pprint
import praw
from praw.models import MoreComments
import requests
import json
import numpy as np
import pandas as pd
from collections import Counter
from math import ceil
from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams

import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.tokenize import word_tokenize

from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten

from math import floor

client_id = '71ZX5Cupn2Ohpg'
client_secret = 'nzCz5_WlQM4LbJxX-t_3m-tPgZw'

reddit = praw.Reddit(user_agent='Comment Extraction',client_id=client_id, client_secret=client_secret)
subreddit = reddit.subreddit('AmITheAsshole')

judgement_categories = ['YTA', 'NTA', 'ESH', 'NAH', 'INFO']

class Submission_Corpus:
    def __init__(self, submission, bfs_depth=2, judgement_categories=None, judgement_weight='upvotes'):
        self.comment_meta = []
        self.judgement_categories = judgement_categories
        self.judgement_weight = judgement_weight
        self.judgements = {}
        for category in self.judgement_categories:
            self.judgements[category] = []

        self.submission = submission
        self.original_post = submission.selftext
        self.comment_forrest = self.submission.comments
        self.comment_bfs(bfs_depth=bfs_depth)       
        
    def comment_bfs(self, bfs_depth=0):
        # Initialize queue to hold comment sub trees during BFS.
        bfs_queue = []
        
        # Populate queue with first level of comments.
        for comment in self.comment_forrest:
            bfs_queue.append(comment)
            
        current_level_size = len(bfs_queue)
        next_level_size = 0
        level_count = 0
        
        while (len(bfs_queue) > 0) and (level_count < bfs_depth):
            comment = bfs_queue.pop(0)
            current_level_size -= 1
            next_level_size += 1
            
            comment_features = None
            try:
                comment_features = self.extract_comment(comment)
            except:
                pass

            if comment_features is not None:
                self.comment_meta.append(comment_features)
                for reply in comment.replies:
                    bfs_queue.append(reply)
            
            if current_level_size == 0:
                current_level_size = next_level_size
                level_count += 1
            
    def extract_comment(self, comment, judgement_extraction_method='prefix', judgement_weighting='upvotes'):
        judgement = self.extract_judgement(comment.body, extraction_method=judgement_extraction_method)
        score = comment.score if judgement_weighting=='upvotes' else 1
        self.judgements[judgement].append(score)
        body = self.tokenize_comment(comment.body)
        comment_features = {
            'id' : comment.id,
            'author': comment.author,
            'body': body,
            'score' : score,
            'judgement': judgement
        }
        return(comment_features)
    
    def extract_judgement(self, txt, extraction_method='prefix'):
        if extraction_method == 'prefix':
            for category in self.judgement_categories:
                if txt[:len(category)] == category:
                    return(category)
    
    def summarize_judgement(self):
        total_judgements = sum([sum(count) for count in self.judgements.values()])
        judgement_summary = [(category, sum(count)/total_judgements) for category, count in self.judgements.items()]
        return(judgement_summary)
    
    def get_judgement_summary(self):
        return(self.judgement_summary)
    
    def tokenize_sentence(self, sent):
        try:
            tokenized_sent = word_tokenize(sent)
        except:
            tokenized_sent = []
        tokenized_sent.insert(0, '<s>')
        tokenized_sent.append('</s>')
        return(tokenized_sent)
    
    def tokenize_comment(self, comment_txt):
        sentences = sent_detector.tokenize(comment_txt.strip())
        tokenized_sentences = [self.tokenize_sentence(sent) for sent in sentences]
        tokenized_comment = ['<c>']
        for sent in tokenized_sentences:
            tokenized_comment += sent
        tokenized_comment += ['</c>']
        return(tokenized_comment)
        
    def get_commentCorpus(self):
        comments_by_category = {}
        for category in self.judgement_categories:
            comments_by_category[category] = []
        
        
        for comment in self.comment_meta:
            if self.judgement_weight == 'upvotes':
                for i in range(max(5,floor(comment['score']/50))):
                    comments_by_category[comment['judgement']].append(comment['body'])
                    
            else:
                comments_by_category[comment['judgement']].append(comment['body'])            
                
        return(comments_by_category)
