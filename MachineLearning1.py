import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import rbm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

#This model as of now is a customizable numerical linear regression model

dataset_path = 'winequality-white.csv'
dataset_sep = ';'
variable_to_solve = 'quality'
testsize = 0.2
randomstate = 42
randomforsetnestimators = 150
bin1 = 2
bin2 = 6.5
bin3 = 9 #8 for red wine 9 for white wine
group_name1 = 'bad'
group_name2 = 'good'
model_sample_prediction_size = 100
mlpc_hidden_layer_size_1 = 207#166
mlpc_hidden_layer_size_2 = 207
mlpc_hidden_layer_size_3 = 207
mlpc_max_iter = 10000 #8000
dataset_header_amount = 10

dataset = pd.read_csv(dataset_path, sep=dataset_sep)

print(dataset.info())
print("")
print("")
print("")
print(dataset.head())
print("")
print("")
print("")
print(dataset.isnull().sum())

bins = (bin1, bin2, bin3)
group_names = [group_name1, group_name2]
dataset[variable_to_solve] = pd.cut(dataset[variable_to_solve], bins=bins, labels=group_names)
print("")
print("")
print("")
print(dataset[variable_to_solve].unique())

label_quality = LabelEncoder()
dataset[variable_to_solve] = label_quality.fit_transform(dataset[variable_to_solve])
print("")
print("")
print("")
print('1 is good wine 0 is bad wine')
print("")
print(dataset.head(dataset_header_amount))
print("")
print("")
print("")
print('this should give you a better idea of the quality of the wine in the full dataset:')
print(dataset[variable_to_solve].value_counts())
print("")
print("")
print("")
print("if I ran this next line in jupyter you would see a bar graph but because I am not you just see this:")
print(sns.countplot(dataset[variable_to_solve]))

X = dataset.drop(variable_to_solve, axis=1)
y = dataset[variable_to_solve]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=randomstate)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("")
print("")
print("")
print("here as bit of the data with standard scalar on it")
print(X_train[:dataset_header_amount])
print("probably can't tell at a glance but this is much better")

#Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=randomforsetnestimators)
rfc.fit(X_train, y_train)
predict_rfc = rfc.predict(X_test)
print("")
print("")
print("")
print("This is a run of the Random Forest Classifier, on some test data")
print(predict_rfc[:model_sample_prediction_size])
print("")
print("Now lets see how this Random Forest Classifier preformed:")
print(classification_report(y_test, predict_rfc))
print("here is our confusion matrix:")
print(confusion_matrix(y_test, predict_rfc))
#Support Vector Classifier
clf=svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)
print("")
print("")
print("")
print("This is a run of the Support Vector Classifier, on some test data")
print(pred_clf[:model_sample_prediction_size])
print("")
print("Now lets see how this Support Vector Classifier preformed:")
print(classification_report(y_test, pred_clf))
print("here is our confusion matrix:")
print(confusion_matrix(y_test, pred_clf))
#Neural Network
mlpc=MLPClassifier(hidden_layer_sizes=(mlpc_hidden_layer_size_1, mlpc_hidden_layer_size_2, mlpc_hidden_layer_size_3), max_iter=mlpc_max_iter)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)
print("")
print("")
print("")
print("This is a run of the Neural Network, or MLPClassifier, on some test data")
print(pred_clf[:model_sample_prediction_size])
print("")
print("Now lets see how this Neural Network, or MLPClassifier, on some test data")
print(classification_report(y_test, pred_mlpc))
print("here is our confusion matrix:")
print(confusion_matrix(y_test, pred_mlpc))
#Stochastic Gradient Descent
sgd=SGDClassifier()
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
print("")
print("")
print("")
print("This is a run of the SGD Classifier (SGD for Stochastic Gradient Descent), on some test data")
print(pred_sgd[:model_sample_prediction_size])
print("")
print("Now lets see how this SGD Classifier or Stochastic Gradient Descent Classifier worked on some test data")
print(classification_report(y_test, pred_sgd))
print("here is our confusion matrix:")
print(confusion_matrix(y_test, pred_sgd))

from sklearn.metrics import accuracy_score
rfc_as = accuracy_score(y_test, predict_rfc)
clf_as = accuracy_score(y_test, pred_clf)
mlpc_as = accuracy_score(y_test, pred_mlpc)
sgd_as = accuracy_score(y_test, pred_sgd)
model_to_use_for_final = "None"
print("")
print("")
print("")
print('RFC:')
print(rfc_as)
print('CLF:')
print(clf_as)
print('MLPC:')
print(mlpc_as)
print("SGD:")
print(sgd_as)
print("")
print("")
if rfc_as > clf_as and rfc_as > mlpc_as and rfc_as > sgd_as:
    print("Looks like the model with the best accuracy is RFC")
    model_to_use_for_final = 'RFC'
elif clf_as > rfc_as and clf_as > mlpc_as and clf_as > sgd_as:
    print("Looks like the model with the best accuracy is CLF")
    model_to_use_for_final = 'CLF'
elif mlpc_as > clf_as and mlpc_as > rfc_as and mlpc_as > sgd_as:
    print("Looks like the model with the best accuracy is MLPC")
    model_to_use_for_final = 'MLPC'
elif sgd_as > clf_as and sgd_as > rfc_as and sgd_as > mlpc_as:
    print("Looks like the model with the best accuracy is SGD")
    model_to_use_for_final = 'SGD'
elif rfc_as == mlpc_as and mlpc_as > sgd_as and rfc_as > sgd_as and mlpc_as > clf_as and rfc_as > clf_as:
    RFCorMLPC = input("The RFC model, and the MLPC model had the same scores, please type either RFC, or MLPC to continue using that model: ")
    if RFCorMLPC == 'RFC':
        print("Okay we will use the RFC model then")
        model_to_use_for_final = 'RFC'
    elif RFCorMLPC == 'MLPC':
        print("Okay we will use the MLPC model then")
        model_to_use_for_final = 'MLPC'
    else:
        print("did not type valid model, will continue with RFC")
        model_to_use_for_final = 'RFC'
elif rfc_as == sgd_as and sgd_as > mlpc_as and rfc_as > mlpc_as and sgd_as > clf_as and rfc_as > clf_as:
    RFCorSGD = input("The RFC model, and the SGD model had the same scores, please type either RFC, or MLPC to continue using that model: ")
    if RFCorSGD == 'RFC':
        print("Okay we will use the RFC model then")
        model_to_use_for_final = 'RFC'
    elif RFCorSGD == 'RFC':
        print("Okay we will use the SGD model then")
        model_to_use_for_final = 'SGD'
    else:
        print("did not type valid model, will continue with RFC")
        model_to_use_for_final = 'RFC'
#Later I need to finish the exception handling. These exceptions are enough for this dataset but others might need more
print("Because of that we will use the " + model_to_use_for_final + " model")
print("")
print("")
print("")
print("The original dataset that I used is pretty high dimensional so I will need too ask you a quite a few questions")
print("")
wine_fixed_acidity = input("What is the fixed acidity of your red wine: ")
wine_volatile_acidity = input("What is the volatile acidity of your red wine: ")
wine_citric_acid = input("What is the citric acidity of your red wine: ")
wine_residual_sugar = input("What is the residual sugar of your red wine: ")
wine_chlorides = input("What is the chlorides of your red wine: ")
wine_free_sulfur_dioxide = input("What is the free sulfur dioxide of your red wine: ")
wine_total_sulfur_dioxide = input("What is the free sulfur dioxide of your red wine: ")
wine_density = input("What is the density of your red wine: ")
wine_pH = input("What is the pH of your red wine: ")
wine_sulphates = input("What is the sulphates of your red wine: ")
wine_alcohol = input("What is the alcohol of your red wine: ")

Xnew = [[wine_fixed_acidity, wine_volatile_acidity, wine_citric_acid, wine_residual_sugar, wine_chlorides, wine_free_sulfur_dioxide, wine_total_sulfur_dioxide, wine_density, wine_pH, wine_sulphates, wine_alcohol]]
Xnew = sc.transform(Xnew)
ynew = None
if model_to_use_for_final == 'RFC':
    ynew = rfc.predict(Xnew)
elif model_to_use_for_final == 'CLF':
    ynew = clf.predict(Xnew)
elif model_to_use_for_final == 'MLPC':
    ynew = mlpc.predict(Xnew)
elif model_to_use_for_final == 'SGD':
    ynew = sgd.predict(Xnew)
print(ynew)