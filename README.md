# Categorised sentence extraction from reviews

Given a set of categories with words that describe them, this code takes in the
text of a product review, and pulls out sentences for each category, returning
the result as a JSON consisting of the sentences relevant to each category, and
a snippet pertinent to each category.

## Overview
Categories are defined as a name (e.g. "price") and a list of descriptor words,
e.g. "price", "cost", "expensive", "inexpensive", "cheap", that a reviewer would
use when discussing the category. It is important to include both positive and
negative descriptors.

The review is split into sentences, and each sentence is tokenized into words,
and the stopwords are removed. Then, we calculate a category score for each
sentence, where the score is defined as the percentage of words of the sentence
that come from that category. Every sentence that has a category score above a
certain threshold is deemed to describe that category. We then return a JSON
that contains, for each category, all the sentences describing that category,
and the sentence with the highest category score. 

## Dependencies
- Uses [NLTK](http://www.nltk.org/) for stemming & tokenization

## Future improvements
- We relying on an individual to define the categories and to describe
them. It would be better to use some sort of embedding, like
[word2vec](https://en.wikipedia.org/wiki/Word2vec) or
[doc2vec](https://arxiv.org/abs/1405.4053) to learn a better representation
of the categories and sentences.  
- If we use some sort of embedding, it would be good to develop our own corpus
of reviews. Currently we do not have that, and would have to use one of the
pre-existing corpuses, which are not from a similar task (they're from, e.g.,
Wikipedia).
- A lot of the system is defined using hand-tuned features. An end-to-end
machine learning classifier would probably perform better.
- Use a better summarisation platform. We have to include redundant words in
the descriptions because NLTK's stemming & lemmatization functions treat
"built" and "build" differently (which is arguably a good thing).
