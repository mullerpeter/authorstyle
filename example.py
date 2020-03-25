from authorstyle import Corpus, average_word_length
from sklearn import metrics

# Load Validation Set and remove class 1
validation_data = Corpus(path='data/pan19-style-change-detection/validation')
validation_data.problems = [problem for problem in validation_data.problems if problem.truth['authors'] > 1]
print('Validation set loaded')

# Perform feature extraction for each sample in the validation set
true = []
pred = []
for problem in validation_data.problems:
    feature = average_word_length(problem.text)

    # Demo prediction method (not really smart)
    num_predicted = int(feature) % 5

    true.append(problem.truth['authors'])
    pred.append(num_predicted)

# Print Validation Score
confusion_matrix = metrics.confusion_matrix(true, pred)
val_accuracy = metrics.accuracy_score(true, pred)

print('Validation Accuracy:', val_accuracy)
