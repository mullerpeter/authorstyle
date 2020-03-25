from .util import *
import os


def punctuation_frequency(text):
    """
   :type text: Text
   :param text: The text to be analysed
   :rtype dict
   :returns A Dictionary mapping the punctuation to their frequency
   """
    punctuations = [',', '.', '?', '!', ':', ';', '’', '"']
    return {c: (text.text.count(c)) / len(text.tokens) for c in punctuations}


def special_character_frequency(text):
    """
   :type text: Text
   :param text: The text to be analysed
   :rtype dict
   :returns A Dictionary mapping the special character to their frequency
   """
    special_chars = ['~', '@', '#', '$', '%', '^', '&', '*', '-', '_', '=', '+', '>', '<', '[', ']', '{', '}', '/',
                     '\\', '|']
    return {c: (text.text.count(c)) / len(text.tokens) for c in special_chars}


def uppercase_frequency(text):
    """
   :type text: Text
   :param text: The text to be analysed
   :rtype float
   :returns Frequency of UpperCase Letter over document length
   """
    return sum(1 for c in text.text if c.isupper()) / len(text.tokens)


def number_frequency(text):
    """
   :type text: Text
   :param text: The text to be analysed
   :rtype float
   :returns Frequency of Numbers over document length
   """
    return sum(1 for c in text.text if c.isdigit()) / len(text.tokens)


def functionword_frequency(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype Dict
    :returns Dict mapping functionwords to frequency per n terms
    """
    with open(os.path.join(os.path.dirname(__file__), 'external_data/functionwords.txt'), 'r') as f:
        functionwords = f.read().splitlines()
    return {functionword: term_frequency(text, functionword, 1) for functionword in functionwords}


def most_common_words_without_stopwords(text, top=50):
    """
    Returns the frequency of the documents top n words in the text
    :type text: Text
    :param text: The text to be analysed
    :type text: Text
    :param top: int
    :type top: Top n n-grams to evaluate
    :returns Returns list of frequencies
    """
    top_max = min([text.document.words_without_stopwords_frequency.N(), top])
    doc_n_grams = text.document.words_without_stopwords_frequency.most_common(top_max)
    n_freq = text.words_without_stopwords_frequency
    feature_vector = []
    for n_gram, freq in doc_n_grams:
        feature_vector.append(0.0 if n_freq.N() == 0 else (n_freq[n_gram] / n_freq.N()))

    if len(feature_vector) != top:
        feature_vector += [0.0] * (top - len(feature_vector))

    return feature_vector


def stopword_ratio(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype Dict
    :returns Returns the ratio of stopwords in the text
    """
    return (len(text.tokens_alphabetic) - len(text.tokens_without_stopwords)) / len(text.tokens_alphabetic)


def top_word_bigram_frequency(text, top=50):
    """
    Returns the frequency of the documents top n word bigrams in the text
    :type text: Text
    :param text: The text to be analysed
    :type text: Text
    :param top: int
    :type top: Top n n-grams to evaluate
    :returns Returns list of frequencies
    """
    top_max = min([text.document.word_bigram_frequency.N(), top])
    doc_n_grams = text.document.word_bigram_frequency.most_common(top_max)
    n_freq = text.word_bigram_frequency
    feature_vector = []
    for n_gram, freq in doc_n_grams:
        feature_vector.append(0.0 if n_freq.N() == 0 else (n_freq[n_gram] / n_freq.N()))

    if len(feature_vector) != top:
        feature_vector += [0.0] * (top - len(feature_vector))

    return feature_vector


def top_bigram_frequency(text, top=50):
    """
    Returns the frequency of the documents top n character bigrams in the text
    :type text: Text
    :param text: The text to be analysed
    :type text: Text
    :param top: int
    :type top: Top n n-grams to evaluate
    :returns Returns list of frequencies
    """
    top_max = min([text.document.bigram_frequency.N(), top])
    doc_n_grams = text.document.bigram_frequency.most_common(top_max)
    n_freq = text.bigram_frequency
    feature_vector = []
    for n_gram, freq in doc_n_grams:
        feature_vector.append(0.0 if n_freq.N() == 0 else (n_freq[n_gram] / n_freq.N()))

    if len(feature_vector) != top:
        feature_vector += [0.0] * (top - len(feature_vector))

    return feature_vector


def top_3_gram_frequency(text, top=50):
    """
    Returns the frequency of the documents top n character 3-grams in the text
    :type text: Text
    :param text: The text to be analysed
    :type text: Text
    :param top: int
    :type top: Top n n-grams to evaluate
    :returns Returns list of frequencies
    """
    doc_n_grams = n_gram_frequency(text.document, n=3)
    top_max = min([doc_n_grams.N(), top])
    doc_n_grams = doc_n_grams.most_common(top_max)
    n_freq = n_gram_frequency(text, n=3)
    feature_vector = []
    for n_gram, freq in doc_n_grams:
        feature_vector.append(0.0 if n_freq.N() == 0 else (n_freq[n_gram] / n_freq.N()))

    if len(feature_vector) != top:
        feature_vector += [0.0] * (top - len(feature_vector))

    return feature_vector
