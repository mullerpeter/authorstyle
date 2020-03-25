import numpy as np
import nltk
import os
import pandas as pd
import collections
from textstat.textstat import textstat
from cophi import complexity


def average_word_length(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype float
    :returns Average number of chars per word
    """
    return len(text.text) / len(text.tokens)


def pos_tag_frequency(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype Dict
    :returns Dict mapping pos tag to frequency per n terms
    """
    pos_tags = ['ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRT', 'PRON', 'VERB', '.', 'X']
    tag_fd = nltk.FreqDist(tag for (word, tag) in text.pos_tags)
    return {tag: (tag_fd[tag] / tag_fd.N()) for tag in pos_tags}


def pos_tag_trigram_frequency(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype Dict
    :returns Dict mapping pos tag to frequency per n terms
    """

    doc_trigrams = text.document.pos_tag_trigram_freq.most_common(20)
    feature_vector = []
    tag_fd = text.pos_tag_trigram_freq
    # return {trigram: (tag_fd[trigram] / tag_fd.N()) for trigram in trigrams}
    for n_gram, freq in doc_trigrams:
        feature_vector.append(0.0 if tag_fd.N() == 0 else (tag_fd[n_gram] / tag_fd.N()))

    return feature_vector


def word_length_distribution(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype list[floats]
    :returns distribution of words from length 1-10 chars
    """
    word_count = [0.0] * 10
    for t in text.tokens_alphabetic:
        if len(t) <= 10:
            word_count[len(t) - 1] += 1
    return [f / len(text.tokens_alphabetic) for f in word_count]


def average_sentence_length_words(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype float
    :returns Average number of words per sentence
    """
    return np.average(text.sentence_length)


def average_syllables_per_word(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype float
    :returns Average syllables per word
    """
    return textstat.syllable_count(text.text) / len(text.tokens_alphabetic)


def average_sentence_length_chars(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype float
    :returns Average number of chars per sentence
    """
    return np.average([len(sentence) for sentence in text.sentences])


def sentence_length_distribution(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype list[float]
    :returns distribution of sentences from length 1-100 words
    """
    sentence_count = [0.0] * 7

    for s in text.sentences:
        if len(s.split()) < 5:
            sentence_count[0] += 1
        elif len(s.split()) < 10:
            sentence_count[1] += 1
        elif len(s.split()) < 15:
            sentence_count[2] += 1
        elif len(s.split()) < 20:
            sentence_count[3] += 1
        elif len(s.split()) < 25:
            sentence_count[4] += 1
        elif len(s.split()) < 30:
            sentence_count[5] += 1
        else:
            sentence_count[6] += 1

    return [f / len(text.sentences) for f in sentence_count]


def yule_k_metric(text):
    """
    :returns Yule K Metric
    :rtype Float
    """
    num_tokens = len(text.tokens_alphabetic)
    bow = collections.Counter(text.tokens_alphabetic)
    freq_spectrum = pd.Series(collections.Counter(bow.values()))

    return complexity.yule_k(num_tokens, freq_spectrum)


def sichel_s_metric(text):
    """
    :returns Sichel S Metric
    :rtype float
    """

    bow = collections.Counter(text.tokens_alphabetic)
    lex_size = len(bow)

    return complexity.sichel_s(lex_size, collections.Counter(bow.values()))


def average_word_frequency_class(text, word_class_mapping=None):
    """
    Average word frequency class as proposed by Sven Meyer zu Eissen, Benno Stein, and Marion Kulig
    :type text: Text
    :param text: The text to be analysed
    :type word_class_mapping: dict
    :param word_class_mapping: Mapping of words to their class
    :rtype float
    :returns The average word frequency class
    """
    if word_class_mapping is None:
        with open(os.path.join(os.path.dirname(__file__), 'external_data/word_class_mapping.txt'), 'r') as file:
            word_class_mapping = eval(file.read())

    return sum([word_class_mapping.get(word, 20) for word in text.tokens_alphabetic]) / max(len(text.tokens_alphabetic),
                                                                                            1)
