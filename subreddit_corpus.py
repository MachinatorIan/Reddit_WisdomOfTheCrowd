import praw
from praw.models import MoreComments
from submission_corpus import SubmissionCorpus

class SubredditCorpus:
    def __init__(self, subreddit=None, retrieval_limit=None, bfs_depth=3, judgement_categories=None):
        self.subreddit = subreddit
        self.labeled_submissions = []
        
        for submission in self.subreddit.new(limit=retrieval_limit):
            submission_corpus = SubmissionCorpus(submission, bfs_depth=bfs_depth, judgement_categories=judgement_categories)
            submission_judgement = [tup[1] for tup in submission_corpus.summarize_judgement()]
            if any(submission_judgement):
                submission_content = (submission.id, submission.selftext)
                self.labeled_submissions.append((submission_content, submission_judgement))
    
    def get_subredditCorpus(self):
        return(self.labeled_submissions)