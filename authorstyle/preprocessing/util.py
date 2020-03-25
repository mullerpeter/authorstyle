from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer


def make_tokens_alphabetic(text):
    """
    Remove all non alphabetic tokens
    :type text: str
    :param text: The text
    :rtype List of str
    :returns List of alphabetic tokens
    """

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())  # This step is needed to prevent hyphenate words from
    # being filtered out
    return [t for t in tokens if t.isalpha()]


def remove_stopwords(tokens, stopwords=nltk_stopwords.words('english')):
    """
    Removes all stopwords from the document tokens
    :type tokens: list of str
    :param tokens: List of tokens
    :type stopwords: list of str
    :param stopwords: List of stopwords to be removed from the document tokens. (Default: Stopword List from nltk)
    :rtype List of str
    :returns List of tokens without stopwords
    """

    return [t for t in tokens if t not in stopwords]


def stem_tokens(tokens):
    """
    Stem all tokens in List
    :type tokens: list of str
    :param tokens: List of tokens
    :rtype List of str
    :returns List of stemmed tokens
    """
    porter_stemmer = PorterStemmer()
    return [porter_stemmer.stem(t) for t in tokens]
