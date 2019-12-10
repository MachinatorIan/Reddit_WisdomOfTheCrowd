import praw
from praw.models import MoreComments
import pandas as pd
from collections import Counter

import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.tokenize import word_tokenize

from math import floor

class SubmissionCorpus:
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
        
        # Store the number of roots in the forrest.
        # (each root is a comment in the initial layer of the comment forrest)
        current_level_size = len(bfs_queue)
        next_level_size = 0
        level_count = 0
        
        # Standard iterative queue-based breadth first search with a depth limit.
        while (len(bfs_queue) > 0) and (level_count < bfs_depth):
            comment = bfs_queue.pop(0)

            # Update level sizes
            # Level size corresponds to the number of nodes of a particular level of the comment forrest
            # (or bfs depth) currently in the queue. When current_level_size = 0, the current depth
            # must be incremented. The current bfs depth is tracked in order to enfore bfs depth limits.
            current_level_size -= 1
            next_level_size += 1
            
            # Extract comment features.
            comment_features = None
            # A try/except is used because the try will almost always succeed.
            # The try only fails when a comment tree is invalid. e.g. a moderator announcement.
            try:
                comment_features = self.extract_comment(comment)
            except:
                pass
            
            # If the comment tree is valid, append the features for future processing, and add replies
            # to bfs queue/
            if comment_features is not None:
                self.comment_meta.append(comment_features)
                for reply in comment.replies:
                    bfs_queue.append(reply)
            
            # If the current level size = 0, no more nodes of the current level exist in the bfs queue,
            # thus, all node currently in the queue are of the next level of the comment forrest. As such
            # the current depth must be updated.
            if current_level_size == 0:
                current_level_size = next_level_size
                level_count += 1
            
    def extract_comment(self, comment, judgement_extraction_method='prefix', judgement_weighting='upvotes'):
        # Extract judgement from comment using the specified method.
        judgement = self.extract_judgement(comment.body, extraction_method=judgement_extraction_method)
        
        # Extract the score using the specified method.
        score = comment.score if judgement_weighting=='upvotes' else 1
        
        # Append the judgement to the list of judgements for the submission. This will be used later to 
        # summarize the judgement of the crowd.
        self.judgements[judgement].append(score)

        # Tokenize the text body of the comment.
        body = self.tokenize_comment(comment.body)

        # Build fature dictionary for comment.
        comment_features = {
            'id' : comment.id,
            'author': comment.author,
            'body': body,
            'score' : score,
            'judgement': judgement
        }

        return(comment_features)
    
    def extract_judgement(self, txt, extraction_method='prefix'):
        # Prefixing is an encoding method whereby judgements are encoded as prefixes to the body of the 
        # comment. This is the most common method used on Reddit.
        if extraction_method == 'prefix':
            # Check for prefix match on each judgement category, and return matching category.
            for category in self.judgement_categories:
                if txt[:len(category)] == category:
                    return(category)
    
    def summarize_judgement(self):
        # Build a simple summary of judgements from the proportion of all comments that vote for each 
        # judgement category.
        total_judgements = sum([sum(count) for count in self.judgements.values()])
        try:
            judgement_summary = [(category, sum(count)/total_judgements) for category, count in self.judgements.items()]
        except:
            judgement_summary = [(0,0)]
        return(judgement_summary)
    
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
        
    def get_commentCorpus(self):
        # Initialize empty dictionary to hold tokenized comments of each judgement category.
        comments_by_category = {}

        # Set the keys of the dictionary to judgement categories, and the initial values as 
        # empty lists.
        for category in self.judgement_categories:
            comments_by_category[category] = []
        
        # Add tokenized comments to the corresponding judgement category list. 
        for comment in self.comment_meta:
            if self.judgement_weight == 'upvotes':
                for _ in range(max(5,floor(comment['score']/50))):
                    comments_by_category[comment['judgement']].append((comment['score'],comment['body']))
                    
            else:
                comments_by_category[comment['judgement']].append((1,comment['body']))            
                
        return(comments_by_category)
