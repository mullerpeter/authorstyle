from textstat.textstat import textstat


def flesch_reading_ease(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype float
    :returns Flesch Reading Ease Score
    """
    return textstat.flesch_reading_ease(text.text)


def automated_readability_index(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype float
    :returns Automated readability index
    """
    return textstat.automated_readability_index(text.text)


def coleman_liau_index(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype float
    :returns Coleman liau index
    """
    return textstat.coleman_liau_index(text.text)


def linsear_write_formula(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype float
    :returns Linsear Write Formula
    """
    return textstat.linsear_write_formula(text.text)


def smog_index(text):
    """
    :type text: Text
    :param text: The text to be analysed
    :rtype float
    :returns Smog Index
    """
    return textstat.smog_index(text.text)
