import random
import os
import numpy as np
import authorstyle

from . import Document, Text
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold


class Problem(object):
    """
    Problem is a wrapper class storing a problem of a corpus
    """

    def __init__(self, file_path=None, problem_name=None, text=None, corpus=None):
        self.name = problem_name
        self.corpus = corpus
        self.text = None
        self.truth = None
        self.id = None
        self.feature_vector = None

        if file_path is not None:
            self.text = Document(file_path)
        if text is not None:
            self.text = Text(text)

    def __eq__(self, other):
        return self.text.text == other.text.text

    def __hash__(self):
        return hash(('text', self.text.text))

    def scale_features(self, scaler=None):
        """
        Scales the feature vectors of the problem
        :type scaler: sklearn Scaler
        :param scaler: Custom scaler to use, default is the corpus scaler
        """
        if scaler is None:
            if self.corpus is None:
                scaler = MinMaxScaler()
                scaler.fit(self.feature_vector)
            else:
                scaler = self.corpus.scaler
        try:
            self.feature_vector = scaler.transform(self.feature_vector)
        except ValueError:
            print("Error on Problem: " + self.name)
            scaler = MinMaxScaler()
            scaler.fit(self.feature_vector)
            self.feature_vector = scaler.transform(self.feature_vector)

    def calculate_feature_vector(self, functions, window_size, window_steps, word_class_mapping=None):
        """
        Calculate feature vector for all problems in corpus
        :type functions: list of feature function
        :param functions: List of feature function to extract from each problem
        :type window_size: int
        :param window_size: Window Size for sliding window
        :type window_steps: int
        :param window_steps: Step Size for sliding window
        :type word_class_mapping: dict
        :param word_class_mapping: loaded word class mapping
        """
        self.text.set_sliding_window(window_size, window_steps)
        self.feature_vector = authorstyle.get_feature_vector(self.text, functions, fragments=True,
                                                             word_class_mapping=word_class_mapping)
        self.text.delete_fragments()
        return self.feature_vector


class Corpus(object):
    """
    Corpus is a wrapper class storing a corpus
    """

    def __init__(self, path=None):
        self.problems = []
        self.scaler = MinMaxScaler()
        self._max_doc_len = None

        if path is not None:
            self.load_corpus(path)

    @property
    def max_doc_len(self):
        if self._max_doc_len is None:
            self._max_doc_len = max([len(problem.text.tokens) for problem in self.problems])
        return self._max_doc_len

    def calculate_feature_vectors(self, functions, window_size, window_steps):
        """
        Calculate feature vector for all problems in corpus
        :type functions: list of feature function
        :param functions: List of feature function to extract from each problem
        :type window_size: int
        :param window_size: Window Size for sliding window
        :type window_steps: int
        :param window_steps: Step Size for sliding window
        """
        if authorstyle.average_word_frequency_class in functions:
            word_class_mapping = authorstyle.get_word_class_mapping()
        else:
            word_class_mapping = None
        for problem in self.problems:
            problem.calculate_feature_vector(functions, window_size, window_steps, word_class_mapping)
            self.scaler.partial_fit(problem.feature_vector)

    def return_kfolds(self, k=5):
        """
        Returns the k folds drawn from the entire corpus
        :type k: int
        :param k: number of folds
        :returns list: list of k (train, validation) sets
        """
        y = [problem.truth['authors'] for problem in self.problems]
        skf = StratifiedKFold(n_splits=k, random_state=54321)
        all_problems = []
        for train_index, test_index in skf.split(np.zeros(len(y)), y):
            fold_problems_train = []
            fold_problems_test = []
            for index in test_index:
                fold_problems_test.append(self.problems[index])
            for index in train_index:
                fold_problems_train.append(self.problems[index])
            all_problems.append((fold_problems_train, fold_problems_test))
        return all_problems

    def scale_features(self, scaler=None):
        """
        Scales the feature vectors of all problems inside the corpus
        :type scaler: sklearn Scaler
        :param scaler: Custom scaler to use, default is the corpus scaler
        """
        scaler = self.scaler if scaler is None else scaler
        for problem in self.problems:
            problem.scale_features(scaler=scaler)

    def load_corpus(self, path):
        """
        Loads the PAN data set from the given directory
        :type path: str
        :param path: file path of the PAN data set
        """
        for file in os.listdir(path):
            if ".txt" in file:
                problem = Problem(path + '/' + file, problem_name=file, corpus=self)
                problem.id = int(file[8:-4])
                problem.truth = eval(open(path + '/' + file[:-3] + 'truth').read(), {'false': False, 'true': True})
                self.problems.append(problem)
        self.problems = sorted(self.problems, key=lambda k: k.id)

    def get_augmented_sample(self, num_authors):
        """
        Returns an augmented sample composed of random segments from the corpus
        :type num_authors: int
        :param num_authors: Number of authors the augmented sample should have
        :returns Problem: Augmented Sample
        """
        text_chunks = []
        structure = []
        weighted_choice = []

        all_labels = [problem.truth['authors'] for problem in self.problems]

        for label in list(set(all_labels)):
            weighted_choice += [label] * (all_labels.count(label) * label)

        for i in range(num_authors):
            weighted_num_author = random.choice(weighted_choice)
            problem = random.choice([problem for problem in self.problems
                                     if problem.truth['authors'] == weighted_num_author])
            random_author = random.randint(1, int(problem.truth['authors']))

            problem_switches = [0] + problem.truth['switches'] + [len(problem.text.text)]
            problem_structure = problem.truth['structure']
            text_chunks += [problem.text.text[problem_switches[i2]:problem_switches[i2 + 1]] for i2 in
                            range(len(problem_structure)) if problem_structure[i2] == ('A' + str(random_author))]
            structure += [('A' + str(i + 1)) for i2 in range(len(problem_structure))
                          if problem_structure[i2] == ('A' + str(random_author))]

        index_list = [x for x in range(len(text_chunks))]
        random.shuffle(index_list)

        text_chunks = list([text_chunks[i] for i in index_list])
        structure = list([structure[i] for i in index_list])
        switches = list([sum([len(text_chunks[i3]) for i3 in range(i2 + 1)]) for i2 in range(len(text_chunks) - 1)])

        augmented_problem = Problem(problem_name='Augmented Problem', text=' '.join(text_chunks))
        augmented_problem.truth = {
            'authors': num_authors,
            'structure': structure,
            'switches': switches
        }

        if len(augmented_problem.text.tokens) > self.max_doc_len:
            return self.get_augmented_sample(num_authors)

        return augmented_problem
