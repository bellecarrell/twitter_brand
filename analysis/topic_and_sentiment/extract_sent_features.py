'''
Use sentiment lexicon to attach sentiment score to each tweet.

Use the following sentiment lexicon:

Sentiment140 Lexicon
@InProceedings{MohammadKZ2013,
  author    = {Mohammad, Saif and Kiritchenko, Svetlana and Zhu, Xiaodan},
  title     = {NRC-Canada: Building the State-of-the-Art in Sentiment Analysis of Tweets},
  booktitle = {Proceedings of the seventh international workshop on Semantic Evaluation Exercises (SemEval-2013)},
  month     = {June},
  year      = {2013},
  address   = {Atlanta, Georgia, USA}
}

Procedure used to tag for tweet sentiment is the same as that used in section 6.2.1 of:

Kiritchenko, S., Zhu, X., Mohammad, S. (2014). Sentiment Analysis of Short Informal Texts.  Journal of Artificial Intelligence Research, 50:723-762, 2014

We use the NRC Emoticon Affirmative and Negated Context Lexicons and take the sum of sentiment scores as
the tweet sentiment.  Features occurring in a negated context (span from a negation word in a closed class
to punctuation or the end of the utterance) are treated separately from an affirmative context (everything else).
'''


def main():
    pass


if __name__ == '__main__':
    pass