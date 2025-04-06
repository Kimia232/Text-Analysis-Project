# PART 1: Harvesting Data (I am using Wikipedia)
from mediawiki import MediaWiki

wikipedia = MediaWiki()
Googoosh = wikipedia.page("Googoosh")
print(Googoosh.title)
print(Googoosh.content)

# assign content to variable
text = Googoosh.content
# PART 2: Analyzing Text
# Text Processing
import string

# Removing Stop words
stop_words = {
    "the",
    "and",
    "a",
    "an",
    "in",
    "of",
    "to",
    "it",
    "is",
    "on",
    "for",
    "with",
    "that",
    "this",
    "was",
    "as",
    "by",
    "at",
    "from",
    "or",
    "are",
    "be",
    "he",
    "she",
    "they",
    "his",
    "her",
    "its",
    "but",
    "have",
    "has",
    "had",
    "not",
    "we",
    "you",
    "i",
    "their",
    "them",
    "who",
    "what",
    "which",
    "when",
    "where",
    "how",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "should",
    "so",
}


# Characterizing by Word Frequencies
def process_text(text):
    """Creates a histogram from a block of text."""
    hist = {}
    strippables = string.punctuation + string.whitespace

    text = text.replace("-", " ")
    text = text.replace("—", " ")
    for word in text.split():
        word = word.strip(strippables).lower()
        if word:
            hist[word] = hist.get(word, 0) + 1
    return hist


# Call the function with the Googoosh Wikipedia content
hist = process_text(text)

# Print the resulting histogram
print(hist)


# Metrics


def total_words(hist):
    """Returns the total of the frequencies in a histogram."""
    return sum(hist.values())

def most_common(hist, excluding_stopwords=False):
    """Makes a list of word-freq pairs in descending order of frequency.

    hist: map from word to frequency

    returns: list of (frequency, word) pairs
    """
    sorted_list = sorted([(freq, word) for word, freq in hist.items()], reverse=True)
    return sorted_list
    
def different_words(hist):
     """Returns the number of different words in a histogram."""
     return len(hist)


def most_common(hist, excluding_stopwords=False):
    """Makes a list of word-freq pairs in descending order of frequency.

    hist: map from word to frequency

    returns: list of (frequency, word) pairs
    """
    if excluding_stopwords:
        hist = {word: freq for word, freq in hist.items() if word not in stop_words}
    sorted_list = sorted([(freq, word) for word, freq in hist.items()], reverse=True)
    return sorted_list


def print_most_common(hist, num=10):
    """Makes a list of word-freq pairs in descending order of frequency.

    hist: map from word to frequency

    returns: list of (frequency, word) pairs
    """
    common = most_common(hist)
    for freq, word in common[:num]:
        print(f"{word:<15} {freq}")


# Activating
hist = process_text(text)

print(f"Total number of words: {total_words(hist)}")
print(f"Number of different words: {different_words(hist)}")

print("\nThe most common words are (excluding stop words):")
t = most_common(hist, excluding_stopwords=True)
for freq, word in t[:20]:
    print(f"{word:<15} {freq}")


def main():

    if __name__ == "__main__":
        main()


# Using Natural Language Processing
import nltk

nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentence = 'She was quickly called "Googoosh", an Armenian name normally exclusively used for boys but which became her stage name.'
score = SentimentIntensityAnalyzer().polarity_scores(sentence)
print(score)
# Output
# {'neg': 0.0, 'neu': 0.614, 'pos': 0.386, 'compound': 0.7417}

# Text Clustering (i attempted it!)
# using other iraninan artists as well
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

from mediawiki import MediaWiki

# Full titles of other artists
titles = ["Googoosh", "Ebi (singer)", "Hayedeh", "Shajarian", "Mohammad Reza Shajarian"]

wikipedia = MediaWiki()
texts = []

for title in titles:
    page = wikipedia.page(title)
    texts.append(page.content)


# these are the similarities computed from the previous section
S = np.asarray(
    [
        [1.0, 0.90850572, 0.96451312, 0.97905034, 0.78340575],
        [0.90850572, 1.0, 0.95769915, 0.95030073, 0.87322494],
        [0.96451312, 0.95769915, 1.0, 0.98230284, 0.83381607],
        [0.97905034, 0.95030073, 0.98230284, 1.0, 0.82953109],
        [0.78340575, 0.87322494, 0.83381607, 0.82953109, 1.0],
    ]
)

# dissimilarity is 1 minus similarity
dissimilarities = 1 - S

# compute the embedding
coord = MDS(dissimilarity="precomputed").fit_transform(dissimilarities)

plt.scatter(coord[:, 0], coord[:, 1])

# Label the points
for i in range(coord.shape[0]):
    plt.annotate(str(i), (coord[i, :]))

plt.show()
