import speech_recognition as sr
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import playsound
import time
import os
from gtts import gTTS
from csv import writer
import pandas as pd
#this program will run a sentiment analysis model that I made based on speech


def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = 'voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ''

        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print('I do not understand')

    return said.lower()
def save_settings_in_csv(name, model, train_x, train_y, test_x, test_y): # working on this
    open(name, mode='w')
    with open(name, mode='a+') as f:
        #df = pd.read_csv(f)
        csv_writer = writer(f)
        #df["model_path"] = model
        csv_writer.writerow(model)
        csv_writer.writerow(train_x)
        csv_writer.writerow(train_y)
        csv_writer.writerow(test_x)
        csv_writer.writerow(test_y)
        #df["train_x"] = train_x
        #df["train_y"] = train_y
        #df["test_x"] = test_x
        #df["test_y"] = test_y
        #df.to_csv(name, index=False)

speak('Hello Thomas')
speak("I'll need you to give input on some things about the model you want to run")
speak("Type the filepath of the .pkl model")
model_file_path = input("Type the file path of the .pkl model:")
speak('Type the file path of the .pkl file that contains the data to fit the Tfidf Vectorizer')
vect_fit_data_path = input("Type the file path of the .pkl file that contains the data to fit the Tfidf Vectorizer:")
Train_y_path = input("Type the file path of the .pkl file that contains your train y data:")
Test_x_path = input("Type the file path of the .pkl file that contains your test_x data: ")
Test_y_path = input("Type the file path of the .pkl file that contains your test_y data:")
print('Would you like to save these paths to a csv?')
speak('Would you like to save these paths to a csv?')
save_to_csv = get_audio()
if 'yes' in save_to_csv:
    print("Okay what do you want to name the file?")
    speak("Okay what do you want to name the file?")
    name = get_audio()
    namestr = name + '.csv'
    name = namestr
    speak("Saving")
    save_settings_in_csv(name, model_file_path, vect_fit_data_path, Train_y_path, Test_x_path, Test_y_path)

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
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything :")
        speak("Say anything")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You said : {}".format(text))
            speak("You said " + text)
        except:
            print("Sorry could not recognize what you said")
            speak("Sorry I could not recognize what you said")
            text = ''
    user_inputs = []
    user_inputs.append(text)
    vectorized_user_inputs = vectorizer.transform(user_inputs)
    prediction = model.predict(vectorized_user_inputs)
    predictionstr = str(prediction)
    if text != '':
        speak(predictionstr)
        print(model.predict(vectorized_user_inputs))
    else:
        print("Variable 'text' came back empty, you either forgot to talk or could not recognize what you said")
        speak("Variable 'text' came back empty, you either forgot to talk or could not recognize what you said")
    user_inputs.clear()
    print("")
    print("Do you want to run this model again?")
    speak('Do you want to run this model again?')
    with sr.Microphone() as source:
        print("Speak Anything :")
        speak("Say anything")
        audio = r.listen(source)
        try:
            run_again = r.recognize_google(audio)
            print("You said : {}".format(run_again))
            speak('You said ' + run_again)
        except:
            print("Sorry could not recognize what you said")
            speak("Sorry could not recognize what you said")
            run_again = ''
    run_again_question_loop = True

    if run_again in ('yes', 'Yes', 'yeah', 'Yeah', 'Keep it open'):
        speak("Running model again")
        run_model = True
    elif run_again in ('no', 'No', 'nah', 'Nah', 'Close it'):
        speak('Closing program')
        run_model = False
    elif run_again == '':
        run_model = False
        print("Invalid command closing program")
        speak("Closing program")


os.remove('voice.mp3')