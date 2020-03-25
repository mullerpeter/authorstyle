import numpy as np


def document_feature_vector_num(document):
    """
    Generating a dictionary with all feature vectors of type int or float
    :type document: Document
    :param document: The Document
    :rtype Dict
    :returns Dictionary mapping the name of the feature to vector
    """

    feature_dict = {k: np.empty(len(document.fragments)) for k, v in document.features.items()
                    if isinstance(v, (int, float))}

    for i, fragment in enumerate(document.fragments):
        for key, value in fragment.features.items():
            if isinstance(value, (int, float)):
                feature_dict[key][i] = value

    return feature_dict


def document_feature_vector_dict(document):
    """
    Generating a dictionary with all feature vectors of type dictionary
    :type document: Document
    :param document: The Document
    :rtype Dict
    :returns Dictionary mapping the name of the feature to vector
    """

    feature_dict = {k: {k1: np.empty(len(document.fragments)) for k1, v1 in v.items()} for k, v
                    in document.features.items() if isinstance(v, dict)}

    for i, fragment in enumerate(document.fragments):
        for key, value in fragment.features.items():
            if isinstance(value, dict):
                for keyI, valueI in value.items():
                    feature_dict[key][keyI][i] = valueI

    return feature_dict


def feature_variance(document):
    """
    Calculate the variance of all features in the document
    :type document: Document
    :param document: The Document
    :rtype Dict
    :returns Dictionary mapping the name of the feature to its variance
    """

    return {k: v.var() for k, v in document_feature_vector_num(document).items()}


def feature_dispersion(document):
    """
    Calculate the index of dispersion of all features in the document
    :type document: Document
    :param document: The Document
    :rtype Dict
    :returns Dictionary mapping the name of the feature to its variance
    """

    return {k: v.var()/v.mean() for k, v in document_feature_vector_num(document).items()}

