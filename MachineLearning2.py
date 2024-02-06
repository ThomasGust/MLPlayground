from sklearn.metrics import f1_score
import numpy as np
import random
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import pickle

#Right now this model is an sentiment anaysis nlp classifier trained on book reviews

class Sentiment:
    NEGATIVE = 'NEGATIVE'
    NEUTRAL = 'NEUTRAL'
    POSITIVE = 'POSITIVE'


class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE


class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews


    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        neutral = list(filter(lambda x: x.sentiment == Sentiment.NEUTRAL, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative+positive_shrunk
        random.shuffle(self.reviews)

import json

dataset_path = 'Books_small_10000.json'

reviews = []

with open(dataset_path) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))




from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size=0.33, random_state=42)


train_container = ReviewContainer(training)
test_container = ReviewContainer(test)


train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)




#support vector machine
from sklearn import svm
print("Support Vector Machine:")
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

print(clf_svm.predict(test_x_vectors))

#logistic regression
print("")
print("")
print("")
print("Logistic Regression:")
from sklearn.linear_model import LogisticRegression
clf_lrc = LogisticRegression(max_iter=16000)
clf_lrc.fit(train_x_vectors, train_y)
print(clf_lrc.predict(test_x_vectors))
#decision tree
print("")
print("")
print("")
print("Decision tree:")
from sklearn.tree import DecisionTreeClassifier
clf_dtc = DecisionTreeClassifier()
clf_dtc.fit(train_x_vectors, train_y)

print(clf_dtc.predict(test_x_vectors))
#MLP or  or multilayer perceptron network
print("")
print("")
print("")
print("Mutilayer perceptron network:")
clf_mlp = MLPClassifier(hidden_layer_sizes=(150, 150, 150), max_iter=150)
print("Created MLPC")
clf_mlp.fit(train_x_vectors, train_y)
print("Fitted MLPC")
print(clf_mlp.predict((test_x_vectors)))
print("")
print("")
print("")
#Mean accuracy of models
svm_mean_accuracy = clf_svm.score(test_x_vectors, test_y)
lrc_mean_accuracy = clf_lrc.score(test_x_vectors, test_y)
dtc_mean_accuracy = clf_dtc.score(test_x_vectors, test_y)
mlp_mean_accuracy = clf_mlp.score(test_x_vectors, test_y)
print("Mean accuracy of Support Vector Machine, or SVM:")
print(svm_mean_accuracy)
print("Mean accuracy of Logistic Regression Classifier, or LRC")
print(lrc_mean_accuracy)
print("Mean accuracy of Decision Tree (Classifier), or DTC:")
print(dtc_mean_accuracy)
print("Mean accuracy of Multilayer perceptron network, or MLP:")
print(mlp_mean_accuracy)
#F1 score of models
print("")
print("")
print("")
svm_f1_score = f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
lrc_f1_score = f1_score(test_y, clf_lrc.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
dtc_f1_score = f1_score(test_y, clf_dtc.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
mlp_f1_score = f1_score(test_y, clf_mlp.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
print("F1 score of Support Vector Machine, or SVM:")
print(svm_f1_score)
print("F1 score of Linear Regression Classifier, or LRC:")
print(lrc_f1_score)
print("F1 score of Decision Tree (Classifier), or DTC:")
print(dtc_f1_score)
print("F1 score of Multilayer perceptron (network), or MLP:")
print(mlp_f1_score)

#Model Tuning
print("")
print("")
print("")
print("Building tuned models")
svm_params = {'kernel': ('linear', 'rbf', 'sigmoid'), 'C': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024), 'decision_function_shape': ('ovr', 'ovo')}
print('Created SVM params')
lrc_params = {'C': (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024), 'max_iter': (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000, 15000, 20000)}
print('Created LRC params')
dtc_params = {'criterion': ('gini', 'entropy'), 'max_features': (None, 'auto', 'sqrt', 'log2')}
print('Created DTC params')
mlp_params = {'hidden_layer_sizes': (100, 150, 200, 250), 'max_iter': (100, 150, 200, 250), 'solver': ('lbfgs', 'sgd', 'adam')}
print('Created MLP params')
print("Created Params for models")
new_clf_svm = svm.SVC()
print("Created new SVM model")
new_clf_lrc = LogisticRegression()
print("Created new LRC model")
new_clf_dtc = DecisionTreeClassifier()
print("Created new DTC model")
new_clf_mlp = MLPClassifier()
print("Created new MLP model")
print("Created new models")
tuned_clf_svm = GridSearchCV(new_clf_svm, svm_params, cv=5)
print("Created tuned SVM model")
tuned_clf_lrc = GridSearchCV(new_clf_lrc, lrc_params, cv=5)
print("Created tuned LRC model")
tuned_clf_dtc = GridSearchCV(new_clf_dtc, dtc_params, cv=5)
print("Created tuned DTC model")
tuned_clf_mlp = GridSearchCV(new_clf_mlp, mlp_params, cv=5)
print("Created tuned MLP model")
print("Created new tuned models")
tuned_clf_svm.fit(train_x_vectors, train_y)
print("Fitted tuned SVM model")
tuned_clf_lrc.fit(train_x_vectors, train_y)
print("Fitted tuned LRC model")
tuned_clf_dtc.fit(train_x_vectors, train_y)
print("Fitted tuned DTC model")
tuned_clf_mlp.fit(train_x_vectors, train_y)
print("Fitted tuned MLP")
print("Fitted tuned models")
tuned_svm_mean_accuracy = tuned_clf_svm.score(test_x_vectors, test_y)
print("Calculated tuned SVM mean accuracy")
tuned_lrc_mean_accuracy = tuned_clf_lrc.score(test_x_vectors, test_y)
print("Calculated tuned LRC mean accuracy")
tuned_dtc_mean_accuracy = tuned_clf_dtc.score(test_x_vectors, test_y)
print("Calculated tuned DTC mean accuracy")
tuned_mlp_mean_accuracy = tuned_clf_mlp.score(test_x_vectors, test_y)
print("Calculated tuned MLP mean accuracy")
print("Calculated mean accuracy of tuned models")
tuned_svm_f1_score = f1_score(test_y, tuned_clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
print("Calculated tuned SVM f1 score")
tuned_lrc_f1_score = f1_score(test_y, tuned_clf_lrc.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
print("Calculated tuned LRC f1 score")
tuned_dtc_f1_score = f1_score(test_y, tuned_clf_dtc.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
print("Calculated tuned DTC f1 score")
tuned_mlp_f1_score = f1_score(test_y, tuned_clf_mlp.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
print("Calculated tuned MLP f1 score")
print("Calculated f1 score of tuned models")
print("Done")
print("")
print("")
print("")
print("Now lets see how the tuned, and un-tuned, models match up")
print("")
print("Mean accuracy of un-tuned SVM:")
print(svm_mean_accuracy)
print("Mean accuracy of tuned SVM:")
print(tuned_svm_mean_accuracy)
print("f1 score fo un-tuned svm:")
print(svm_f1_score)
print("f1 score of tuned SVM")
print(tuned_svm_f1_score)
print("Mean accuracy of un-tuned LRC:")
print(lrc_mean_accuracy)
print("Mean accuracy of tuned LRC:")
print(tuned_lrc_mean_accuracy)
print("f1 score of un-tuned LRC:")
print(lrc_f1_score)
print("f1 score of tuned LRC:")
print(tuned_lrc_f1_score)
print("Mean accuracy of un-tuned DTC:")
print(dtc_mean_accuracy)
print("Mean accuracy of tuned DTC:")
print(tuned_dtc_mean_accuracy)
print("f1 score of un-tuned DTC:")
print(dtc_f1_score)
print("f1 score of tuned DTC:")
print(tuned_dtc_f1_score)
print("Mean accuracy of un-tuned MLP:")
print(mlp_mean_accuracy)
print("Mean accuracy of tuned MLP: ")
print(tuned_mlp_mean_accuracy)
print("f1 score of un-tuned MLP:")
print(mlp_f1_score)
print("f1 score of tuned MLP:")
print(tuned_mlp_f1_score)
print("")
print("")
print("")
model_type = None
model_selection_metric = None
model_selection_metric_input = input("What accuracy metric do you want to use for model selection. Mean accuracy or f1 score: ")
if model_selection_metric_input == 'Mean accuracy':
    print("Okay we will use the Mean accuracy metric then")
    model_selection_metric = 'Mean accuracy'
elif model_selection_metric_input == 'f1 score':
    print("Okay we will use the f1 score accuracy metric then")
    model_selection_metric = 'f1 score'
else:
    print("Did not type correct valid value. will continue with mean accuracy")
    model_selection_metric = 'Mean accuracy'

if model_selection_metric == 'Mean accuracy':
    if tuned_svm_mean_accuracy > tuned_dtc_mean_accuracy and tuned_svm_mean_accuracy > tuned_lrc_mean_accuracy and tuned_svm_mean_accuracy > tuned_mlp_mean_accuracy:
        print("SVM had the highest mean accuracy so we will use SVM")
        model_type = 'SVM'
    elif tuned_lrc_mean_accuracy > tuned_svm_mean_accuracy and tuned_lrc_mean_accuracy > tuned_dtc_mean_accuracy and tuned_lrc_mean_accuracy > tuned_mlp_mean_accuracy:
        print("LRC had the highest mean accuracy so we will use LRC")
        model_type = 'LRC'
    elif tuned_dtc_mean_accuracy > tuned_svm_mean_accuracy and tuned_dtc_mean_accuracy > tuned_lrc_mean_accuracy and tuned_dtc_mean_accuracy > tuned_mlp_mean_accuracy:
        print("DTC had the highest mean accuracy so we will use DTC")
        model_type = 'DTC'
    elif tuned_mlp_mean_accuracy > tuned_svm_mean_accuracy and tuned_mlp_mean_accuracy > tuned_lrc_mean_accuracy and tuned_mlp_mean_accuracy > tuned_dtc_mean_accuracy:
        print('MLP had the highest mean accuracy so we will use MLP')
        model_type = 'MLP'

elif model_selection_metric == 'f1 score':
    if tuned_svm_f1_score > tuned_lrc_f1_score and tuned_svm_f1_score > tuned_dtc_f1_score and tuned_svm_f1_score > tuned_mlp_f1_score:
        print("SVM had the highest f1 score so we will use SVM")
        model_type = 'SVM'
    elif tuned_lrc_f1_score > tuned_svm_f1_score and tuned_lrc_f1_score > tuned_dtc_f1_score and tuned_lrc_f1_score > tuned_mlp_f1_score:
        print("LRC had the highest f1 score so we will use LRC")
        model_type = 'LRC'
    elif tuned_dtc_f1_score > tuned_svm_f1_score and tuned_dtc_f1_score > tuned_lrc_f1_score and tuned_dtc_f1_score > tuned_mlp_f1_score:
        print("DTC had the highest f1 score so we will use DTC")
        model_type = 'DTC'
    elif tuned_mlp_f1_score > tuned_svm_f1_score and tuned_mlp_f1_score > tuned_lrc_f1_score and tuned_mlp_f1_score > tuned_dtc_f1_score:
        print("MLP had the highest f1 score so we will use MLP")

book_reviews = []
book_review = input("Enter your book review: ")
book_reviews.append(book_review)
vectorized_book_reviews = vectorizer.transform(book_reviews)
if model_type == 'SVM':
    print(tuned_clf_svm.predict(vectorized_book_reviews))
    print("Predicted with tuned SVM")
elif model_type == 'LRC':
    print(tuned_clf_lrc.predict(vectorized_book_reviews))
    print("Predicted with tuned LRC")
elif model_type == 'DTC':
    print(tuned_clf_dtc.predict(vectorized_book_reviews))
    print("Predicted with tuned DTC")
elif model_type == 'MLP':
    print(tuned_clf_mlp.predict(vectorized_book_reviews))
    print("Predicted with tuned MLP")

print("")
print("")
print("")
write_model_to_file = input("Would you like to save this model to a .pkl file so you don't need to retrain it? (yes or no):")
if write_model_to_file == 'yes':
    with open("Saved models/nlp/Sentiment Analysis/1/Train_x.pkl", 'wb') as f:
        f.flush()
        pickle.dump(train_x, f)
    with open("Saved models/nlp/Sentiment Analysis/1/Train_y.pkl", 'wb') as f:
        f.flush()
        pickle.dump(train_y, f)
    with open("Saved models/nlp/Sentiment Analysis/1/Test_x.pkl", 'wb') as f:
        f.flush()
        pickle.dump(test_x, f)
    with open("Saved models/nlp/Sentiment Analysis/1/Test_y.pkl", 'wb') as f:
        f.flush()
        pickle.dump(test_y, f)
    save_file_path = input("Type save directory, and after that type file name (dir_path/file_name.pkl):")
    with open('Saved models/nlp/Sentiment Analysis/1/Sentiment_Classifier_First_Iteration.pkl', 'wb') as f:
        f.flush()
        if model_type == 'SVM':
            pickle.dump(tuned_clf_svm, f)
            print("")
            print("Wrote SVM model to file")
        elif model_type == 'LRC':
            pickle.dump(tuned_clf_lrc, f)
            print("")
            print("Wrote LRC model to file")
        elif model_type == 'DTC':
            pickle.dump(tuned_clf_dtc, f)
            print("")
            print("Wrote DTC model to file")
        elif model_type == 'MLP':
            pickle.dump(tuned_clf_mlp, f)
            print("")
            print("Wrote MLP model to file")