import numpy as np
from nltk import ngrams
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

import authorstyle
import authorstyle.preprocessing as preprocessing


#################################################################
# Text
#################################################################


class Text(object):
    """
    Text is a wrapper class storing a text
    """

    def __init__(self, text):
        """
        Construct a new Document from the given file path
        :type text: string
        :param text: The text to be analysed
        """
        self._text = text
        self._text_preprocessed = None
        self._tokens = None
        self._bigram_frequency = None
        self._bigram_idfs = None
        self._pos_tags = None
        self._pos_tag_trigram_freq = None
        self._tokens_alphabetic = None
        self._tokens_without_stopwords = None
        self._tokens_stemmed = None
        self._sentences = None
        self._sentence_length = None
        self._frequency_distribution = None
        self._word_bigram_frequency = None
        self._words_without_stopwords_frequency = None

        self._features = None
        self._fragments = None

        self.document = self

    @property
    def text(self):
        return self._text

    @property
    def text_preprocessed(self):
        if self._text_preprocessed is None:
            self._text_preprocessed = self._text.encode('ascii', errors='ignore').decode('ascii', errors='ignore') \
                .lower()
        return self._text_preprocessed

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = word_tokenize(self._text.lower())
        return self._tokens

    @property
    def pos_tags(self):
        if self._pos_tags is None:
            self._pos_tags = pos_tag(self.tokens, tagset='universal')
        return self._pos_tags

    @property
    def bigram_frequency(self):
        if self._bigram_frequency is None:
            self._bigram_frequency = FreqDist(ngrams(self.text, 2))
        return self._bigram_frequency

    @property
    def word_bigram_frequency(self):
        if self._word_bigram_frequency is None:
            self._word_bigram_frequency = FreqDist(ngrams(self.tokens_alphabetic, 2))
        return self._word_bigram_frequency

    @property
    def words_without_stopwords_frequency(self):
        if self._words_without_stopwords_frequency is None:
            self._words_without_stopwords_frequency = FreqDist(self.tokens_without_stopwords)
        return self._words_without_stopwords_frequency

    @property
    def bigram_idfs(self):
        if self._bigram_idfs is None:
            doc_n_grams = self.bigram_frequency.most_common(1000)
            idfs = {}
            for n_gram, freq in doc_n_grams:
                idfs[n_gram] = np.log((len(self.fragments)) / (1 + sum(
                    [1 for i in range(len(self.fragments)) if self.fragments[i].bigram_frequency[n_gram] > 0])))
            self._bigram_idfs = idfs
        return self._bigram_idfs

    @property
    def pos_tag_trigram_freq(self):
        if self._pos_tag_trigram_freq is None:
            self._pos_tag_trigram_freq = FreqDist(ngrams([tag for (word, tag) in self.pos_tags], 3))
        return self._pos_tag_trigram_freq

    @property
    def tokens_alphabetic(self):
        if self._tokens_alphabetic is None:
            self._tokens_alphabetic = preprocessing.make_tokens_alphabetic(self._text.lower())
        return self._tokens_alphabetic

    @property
    def tokens_without_stopwords(self):
        if self._tokens_without_stopwords is None:
            self._tokens_without_stopwords = preprocessing.remove_stopwords(self.tokens_alphabetic)
        return self._tokens_without_stopwords

    @property
    def tokens_stemmed(self):
        if self._tokens_stemmed is None:
            self._tokens_stemmed = preprocessing.stem_tokens(self.tokens_without_stopwords)
        return self._tokens_stemmed

    @property
    def sentences(self):
        if self._sentences is None:
            self._sentences = sent_tokenize(self._text)
        return self._sentences

    @property
    def sentence_length(self):
        if self._sentence_length is None:
            self._sentence_length = [len(sentences.split()) for sentences in self.sentences]
        return self._sentence_length

    @property
    def frequency_distribution(self):
        if self._frequency_distribution is None:
            self._frequency_distribution = FreqDist(self.tokens_alphabetic)
        return self._frequency_distribution

    @property
    def features(self):
        if self._features is None:
            self._features = authorstyle.get_feature_vector(self)
        return self._features

    @property
    def fragments(self):
        if self._fragments is None:
            self.set_sliding_window()
        return self._fragments

    def delete_fragments(self):
        self._fragments = None

    def set_sliding_window(self, window_size=50, step_size=10, unit=None):
        """
        Divides the document into chunks of sentences
        :type window_size: int
        :param window_size: Window size defining the number of sentences per chunk
        :type step_size: int
        :param step_size: Step defining how many sentences to move the window per chunk
        :type unit: List
        :param step_size: List Attribute of 'Text' on which the sliding window should be performed
        """
        if unit is None:
            unit = self.sentences
        self._fragments = []
        for i in range(0, len(unit) - 1, step_size):
            if i + window_size < len(unit):
                if len(preprocessing.make_tokens_alphabetic((" ".join(unit[i:i + window_size])).lower())) != 0:
                    fragment = Text(" ".join(unit[i:i + window_size]))
                    fragment.document = self
                    self._fragments.append(fragment)
            else:
                fragment = Text(" ".join(unit[i:]))
                fragment.document = self
                self._fragments.append(fragment)


#################################################################
# Document
#################################################################

class Document(Text):
    """
    Document is a subclass of Text storing and reading a whole document
    """

    def __init__(self, file_path):
        """
        Construct a new Document from the given file path
        :type file_path: string
        :param file_path: The path of the file
        """

        Text.__init__(self, open(file_path, "r", errors='ignore').read())
        self._path = file_path
