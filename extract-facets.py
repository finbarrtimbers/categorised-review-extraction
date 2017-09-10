from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import jaccard_similarity_score

def match(sentence, description):
    # If this proves to be computationally expensive, I'd cache this call.
    # However, I think it's just a dictionary lookup, so it should be cheap.
    stop=set(stopwords.words('english'))
    sentence_words = set([w for w in word_tokenize(sentence) if w
                          not in stop])
    # we don't need to tokenize the description as it's already just words
    description_words = set([w for w in description if w not in stop])
    # This has a number of drawbacks, e.g. if it only has one word... I'd want
    # to think about this more before putting it in production
    # We calculate the % of words in the sentence that fall in the description
    # Drawback: ignores order
    similarity = len(sentence_words & description_words) / len(sentence_words)
    return similarity 
    
def summarise(sentences):
    # N.B.: this is very incomplete. If building a summariser for production,
    # I'd implement something similar to
    # https://github.com/tensorflow/models/tree/master/textsum, probably 
    # whatever is the state of the art from0
    # https://summarization2017.github.io/
    # Right now, what we're doing is picking the sentence with the highest
    # category score
    max_score_index = 0
    for i in range(len(sentences)):
        if sentences[i][1] > sentences[max_score_index][1]:
            max_score_index = i
    return sentences[i][0]
       
def extract_facets(review_text, categories):
    review_sentences = sent_tokenize(review_text)
    categorised_sentences = defaultdict(list)
    THRESHOLD = 0.5
    for sentence in review_sentences:
        # If using tensorflow (or other ML framework), I would do
        # the next step simultaneously, by learning a probability distribution
        # over the N categories and adding the sentences to the category
        # lists if the probability was over a threshold
        for category, category_description in categories:
            match_score = match(sentence, category_description)
            if match_score > THRESHOLD:
                categorised_sentences[category].append((sentence, match_score))
    review_facts = {category: summarise(categorised_sentences[category]) for
                    category in categorised_sentences}
    return json.dumps(review_facts)

def main():
    categories = {'price': ['price', 'cost', 'expensive', 'inexpensive',
                            'cheap'],
                  'durability': ['durable', 'build', 'built', 'lasts', 'breaks',
                                 'broken'],
                  'sound_quality': ['sounds', 'bass', 'treble', 'frequencies',
                                    'fidelity']}
    with open('corpus.txt', 'r') as review_file:
        for review in review_file:
            print(extract_facets(review, categories))
