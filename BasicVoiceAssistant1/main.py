import speech_recognition as sr
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import playsound
import time
import os
from gtts import gTTS
import subprocess
import datetime
import webbrowser

chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'

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


def open_blender():
    subprocess.Popen(['C:\Program Files\Blender Foundation\Blender 2.82' + '/' + 'blender.exe'])
def open_notepad():
    subprocess.Popen(['Notepad.exe'])
def note_notepad_note(text, name):
    date = datetime.datetime.now()
    file_name = str(date).replace(':', '-') + '-note-' + name + '.txt'
    with open(file_name, 'w') as f:
        f.write(text)

    subprocess.Popen(['Notepad.exe', file_name])
def open_school():
    website = 'https://classroom.google.com/c/NTMxMTc3NTc5MjVa'
    webbrowser.get(chrome_path).open_new_tab(website)
def open_youtube():
    website = 'https://youtube.com'
    webbrowser.get(chrome_path).open_new_tab(website)
def open_com_web(com):
    website = 'https://www.' + com + '.com'
    webbrowser.get(chrome_path).open_new_tab(website)
def open_org_web(org):
    website = 'https://www.' + org + '.org'
    webbrowser.get(chrome_path).open_new_tab(website)
def open_net_web(net):
    website = 'https://www.' + net + '.net'
    webbrowser.get(chrome_path).open_new_tab(website)
def open_creative_cloud():
    subprocess.Popen('C:\Program Files\Adobe\Adobe Creative Cloud\ACC\Creative Cloud.exe')
def open_photoshop():
    subprocess.Popen('C:\Program Files\Adobe\Adobe Photoshop 2020\Photoshop.exe')
def open_powerpoing():
    subprocess.Popen("C:\Program Files\Microsoft Office" + "/" + "root\Office16\POWERPNT.EXE")
def open_amazon():
    website = 'https://amazon.com/'
    webbrowser.get(chrome_path).open_new_tab(website)
WAKE = 'hey doug'
on = True

while on == True:
    print('Listening')
    text = get_audio()

    if WAKE in text:
        speak('I am ready')
        text = get_audio()
        if 'open blender' in text:
            open_blender()
        if 'open notepad' in text:
            open_notepad()
        if 'create a note' in text:
            speak('sure. what do you want to say')
            text = get_audio()
            name = get_audio()
            note_notepad_note(text, name=name)
        if 'open school' in text:
            speak('okay')
            open_school()
        if 'open youtube' in text:
            speak('okay')
            open_youtube()
        if 'open cam' in text:
            speak('sure')
            speak('what company do you want to open?')
            com = get_audio()
            speak('okay')
            open_com_web(com=com)
        if 'open net' in text:
            speak('sure')
            speak('what network do you want to open?')
            net = get_audio()
            speak('okay')
            open_net_web(net=net)
        if 'open org' in text:
            speak('sure')
            speak('what network do you want to open?')
            org = get_audio()
            speak('okay')
            open_org_web(org=org)
        if 'open creative cloud' in text:
            speak('sure')
            open_creative_cloud()
        if 'open photoshop' in text:
            speak('sure')
            open_photoshop()
        if 'open powerpoint' in text:
            speak('sure')
            open_powerpoing()
        if 'doug off' in text:
            print('Closing doug')
            speak('Closing doug')
            on = False
