import praw
from praw.models import MoreComments
from submission_corpus import SubmissionCorpus

class SubredditCorpus:
    def __init__(self, subreddit=None, retrieval_limit=None, bfs_depth=3, judgement_categories=None):
        self.subreddit = subreddit
        self.labeled_submissions = []
        self.judgement_categories = judgement_categories
        
        for submission in self.subreddit.new(limit=retrieval_limit):
            submission_corpus = SubmissionCorpus(submission, bfs_depth=bfs_depth, judgement_categories=judgement_categories)
            submission_judgement = [tup[1] for tup in submission_corpus.summarize_judgement()]
            if any(submission_judgement):
                submission_content = (submission.id, submission.selftext)
                self.labeled_submissions.append((submission_content, submission_judgement))
    
    def get_majority(self, judgement_vec):
        majority_judgement = self.judgement_categories[judgement_vec.index(max(judgement_vec))]
        return(majority_judgement)

    def get_subredditCorpus(self):
        sorted_submissions = dict(zip(self.judgement_categories, [[] for _ in range(len(self.judgement_categories))]))
        for labeled_submission in self.labeled_submissions:
            majority_judgement = self.get_majority(labeled_submission[1])
            sorted_submissions[majority_judgement].append(labeled_submission[0])
        for category in self.judgement_categories:
            f = open(category+'.txt', 'w+')
            for i, submission in enumerate(sorted_submissions[category]):
                f.write(str(submission[0])+'\t'+str(submission[1]))
                if i < len(sorted_submissions[category])-2:
                    f.write('\n')
            f.close()

        return(self.labeled_submissions)