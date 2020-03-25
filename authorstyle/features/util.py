import nltk


def term_frequency(text, term, per=1000):
    """
    :type text: Text
    :param text: The text to be analysed
    :param term: A term/word
    :type term: str
    :param per: An Int determining per how many terms the frequency should be evaluated
    :type per: int
    :return: The frequency of the term appearing in the text
    :rtype: float
    """

    return (text.frequency_distribution[term] * per) / text.frequency_distribution.N()


def n_gram_frequency(text, n=2):
    """
    :type text: Text
    :param text: The text to be analysed
    :param n: Defines the character length of the n-grams (Default: 2, bi-grams)
    :type n: int
    :return: The frequency distribution of the n-grams
    :rtype: nltk FreqDist
    """

    ngrams = nltk.ngrams(text.text, n)
    return nltk.FreqDist(ngrams)

