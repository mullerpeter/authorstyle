import authorstyle.features as features
import os


def all_feature_functions():
    """
    Returns all feature functions from the function module
    :rtype list[callable]
    :returns List of feature functions
    """
    exclude = ['n_gram_frequency', 'term_frequency']
    functions = []
    for name in dir(features):
        feature_function = getattr(features, name)
        if callable(feature_function) and feature_function.__name__ not in exclude:
            functions.append(feature_function)
    return functions


def get_word_class_mapping():
    """
    Loads and returns the word-class mapping
    :rtype dict
    :returns Dictionary mapping words to word-class
    """
    with open(os.path.join(os.path.dirname(__file__), 'features/external_data/word_class_mapping.txt'), 'r') as file:
        word_class_mapping = eval(file.read())
    return word_class_mapping


def get_feature_vector(text, functions=None, fragments=False, word_class_mapping=None):
    """
    Calculate the feature vector
    :type text: Text
    :param text: The text to be analyzed
    :type functions: list[callable[Text]]
    :param functions: List of feature functions
    :type fragments: bool
    :param fragments: Should evaluate on every fragment, default=False
    :type word_class_mapping: dict
    :param word_class_mapping: Pre loaded word-class dictionary
    :rtype list[float] or list[list[float]]
    :returns Feature vector (default 1D, 2D when fragments=True)
    """
    if functions is None:
        functions = all_feature_functions()

    vector = []
    if fragments:
        for fragment in text.fragments:
            fragment_vector = []
            if len(fragment.tokens_alphabetic) == 0:
                vector.append(vector[-1])
            else:
                for f in functions:
                    if f.__name__ == 'average_word_frequency_class':
                        if word_class_mapping is None:
                            with open(os.path.join(os.path.dirname(__file__),
                                                   'features/external_data/word_class_mapping.txt'), 'r') as file:
                                word_class_mapping = eval(file.read())
                        value = f(fragment, word_class_mapping=word_class_mapping)
                    else:
                        value = f(fragment)
                    if isinstance(value, list):
                        fragment_vector += value
                    elif isinstance(value, dict):
                        fragment_vector += value.values()
                    else:
                        fragment_vector.append(value)
                vector.append(fragment_vector)
    else:
        for f in functions:
            if f.__name__ == 'average_word_frequency_class':
                if word_class_mapping is None:
                    with open(os.path.join(os.path.dirname(__file__), 'features/external_data/word_class_mapping.txt'),
                              'r') as file:
                        word_class_mapping = eval(file.read())
                value = f(text, word_class_mapping=word_class_mapping)
            else:
                value = f(text)
            if isinstance(value, list):
                vector += value
            elif isinstance(value, dict):
                vector += value.values()
            else:
                vector.append(value)
    return vector
