import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

model_file_path = input("Type the file path of the .pkl model:")
vect_fit_data_path = input("Type the file path of the .pkl file that contains the data to fit the Tfidf Vectorizer:")
Train_y_path = input("Type the file path of the .pkl file that contains your train y data:")
Test_x_path = input("Type the file path of the .pkl file that contains your test_x data: ")
Test_y_path = input("Type the file path of the .pkl file that contains your test_y data:")

with open(file=model_file_path, mode='rb') as f:
    model = pickle.load(f)
with open(file=vect_fit_data_path, mode='rb') as f:
    train_x = pickle.load(f)
with open(Train_y_path, mode='rb') as f:
    train_y = pickle.load(f)
with open(Test_x_path, mode='rb') as f:
    test_x = pickle.load(f)
with open(Test_y_path, mode='rb') as f:
    test_y = pickle.load(f)

print("")
print("Loaded model, and Test/Train data")



vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)
print("")
print("")
print("In case you need a reminder here is the mean accuracy of the model")
model_mean_accuracy = model.score(test_x_vectors, test_y)
print("Model mean accuracy:")
print(model_mean_accuracy)


run_model = True

while run_model == True:
    user_inputs = []
    user_input = input("Enter your text:")
    user_inputs.append(user_input)
    vectorized_user_inputs = vectorizer.transform(user_inputs)
    print(model.predict(vectorized_user_inputs))
    user_inputs.clear()
    print("")
    run_again = input("Do you want to run this model again. (yes or no):")
    if run_again in ('yes', 'Yes'):
        run_model = True
    elif run_again in ('no', 'No'):
        run_model = False
    else:
        print("Invalid text. Closing program")
        run_model = False