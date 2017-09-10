# Categorised sentence extraction from reviews



Spend ~5 minutes browsing customer reviews of Musical Instruments on Amazon (https://www.amazon.com/b?&node=11091801).
Brainstorm an approach that identifies, given a customer review, whether the review mentions one or more of the following product facets: price, durability, and/or sound quality.
Submit a function that takes the raw text of a customer review as input and outputs a JSON consisting of the facets extracted from the review and a snippet pertinent to each facet.

## Overview
- Uses [GloVe](https://github.com/stanfordnlp/GloVe) to create word vectors.
- Uses [NLTK](http://www.nltk.org/) for lemmatization

# Example

If you wish, you may leverage an external knowledge-base (such as WordNet or word embeddings), but please do not use a machine learning framework or toolkit for this task. We are mainly evaluating problem solving and coding.


We are looking for a "reasonable" solution: Does the code use sensible data structures and control flow? Does the approach perform reasonably well given the constraints? How does the solution ensure accuracy when extracting facets and snippets?

Does the code have to build?

The code does not have to build, but it should be written in a real programming language (not pseudocode). This ensures that we are able to follow along without guesswork.

## Customisation
- Train the GloVe embeddings on your own corpus by 


## Future improvements
- Get a better corpus
- Use a better summarisation platform
