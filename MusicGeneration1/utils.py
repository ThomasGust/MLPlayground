import math
import random
import midi
from mido import MidiFile, MidiTrack
import os
from os import listdir
import midi
import string
from datetime import datetime
from MusicGeneration1.markov_model_new import MarkovModel
import pickle
from pathlib import Path

# this file contains as of now the core functionality for the music generation project
# having in it the classes 'GenerateMidi', and 'SongGenBuilder', as well as more.
# as this project expand I think most of the functionality will still be in this file


# right now I am still working on the documentation for all of the utilities

def generate_random_song_key():
    starter_val = random.uniform(1, 10)
    square_val_1 = 2
    mul_val_1 = random.uniform(2, 2)
    mul_val_2 = random.uniform(2, 2)
    square_val_2 = 2
    num_1 = ((math.pow((math.pow(starter_val, square_val_1)) * mul_val_1 * mul_val_2, square_val_2))) / 10
    if num_1 < 100:
        num_1 = num_1 * 10
    elif num_1 > 10000:
        num_1 = num_1 / 90
    elif num_1 > 9000:
        num_1 = num_1 / 80
    elif num_1 > 8000:
        num_1 = num_1 / 70
    elif num_1 > 7000:
        num_1 = num_1 / 60
    elif num_1 > 6000:
        num_1 = num_1 / 50
    elif num_1 > 5000:
        num_1 = num_1 / 40
    elif num_1 > 4000:
        num_1 = num_1 / 30
    elif num_1 > 3000:
        num_1 = num_1 / 20
    elif num_1 > 2000:
        num_1 = num_1 / 10
    elif num_1 > 1000:
        num_1 = num_1 / 5
    rounded_num_1 = math.ceil(num_1)
    return rounded_num_1


def read_tracks(midi):
    midi = MidiFile(midi)

    for track in midi.tracks:
        print(track)


# read_tracks('training_music/single_track/Ludwig van Beethoven/appass_1_format0.mid')

def remove_duplicate_tracks(midi):
    midi_obj = MidiFile(midi)

    message_numbers = []
    duplicates = []

    for track in midi_obj.tracks:
        if len(track) in message_numbers:
            duplicates.append(track)
        else:
            message_numbers.append(len(track))

    for track in duplicates:
        midi_obj.tracks.remove(track)

    midi_obj.save(midi)


def alt_midi_type(midi, type):
    midi_obj = MidiFile(midi, type=type)
    return midi_obj


def alt_midi_type_file(midi, type, newfilename, keepoldvariant=True):
    midi_obj = MidiFile(midi, type=type)
    if keepoldvariant == True:
        midi_obj.save(newfilename)
    elif keepoldvariant == False:
        midi_obj.save(midi)


def crash():
    crashingprogramhaha


def midi_to_wav(midi_fn, type=0):
    # this one does not work yet
    midi_obj = MidiFile(midi_fn, type=type)
    midi_to_wav_name = midi_fn.replace('.mid', '.wav')
    midi_obj.save(midi_to_wav_name)


def get_num_tracks(midi):
    midi_obj = MidiFile(midi)

    i = 0

    for track in midi_obj.tracks:
        i += 1
    return i


def randomString(stringLength=30):
    letters = string.ascii_lowercase + string.digits + string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def get_time():
    now = datetime.now()

    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time


def write_to_log(log_path, msg):
    with open(log_path, mode='w') as f:
        f.write(msg)


class MidiGenerator():
    def __init__(self, mkvorder, songname, songnumber, musician, outputdir):
        self.mkvorder = mkvorder
        self.songname = songname
        self.songnumber = songnumber
        self.musician = musician
        self.outputdir = outputdir
        self.fullsongpath = str(self.outputdir) + '/' + str(self.songname) + str(self.songnumber) + '.mid'
        self.corpusfiles = None
        self.messages = None
        self.songfullname = str(self.songname) + str(self.songnumber) + '.mid'

    def get_mkv_order(self):
        return self.mkvorder

    def get_song_name(self):
        return self.mkvorder

    def get_song_number(self):
        return self.songnumber

    def get_musician(self):
        return self.musician

    def get_output_dir(self):
        return self.outputdir

    def set_mkv_order(self, mkvorder):
        self.mkvorder = mkvorder
        return self.mkvorder

    def set_song_name(self, songname):
        self.songname = songname
        return self.songname

    def set_song_number(self, song_number):
        self.songnumber = song_number
        return self.songnumber

    def set_musician(self, musician):
        self.musician = musician
        return self.musician

    def set_output_dir(self, outputdir):
        self.outputdir = outputdir

    def generate_midi(self):
        if self.musician in ('Chopin', 'chopin', 'CHOPIN'):
            print(f'generating song with mkv order {self.mkvorder}, and and a musician of type {self.musician}')
            self.corpusfiles = (MidiFile(f'training_music/single_track/Frédéric Chopin/{file_name}') for file_name in
                                listdir('training_music/single_track/Frédéric Chopin'))
            print('built corpus files')

            self.messages = (message for f in self.corpusfiles for message in
                             ['START'] + f.tracks[len(f.tracks) - 1] + ['END'])
            print('built messages')

            mkv = MarkovModel(midi_data=self.messages, order=self.mkvorder)
            print('Built Markov')

            gen_midi = MidiFile(type=1)
            track = MidiTrack()
            gen_midi.tracks.append(track)

            for self.message in mkv.sample():
                track.append(self.message)
                print(f'appended {self.message} to track')

            gen_midi.save(self.fullsongpath)
            print(
                f'created and, saved {self.songfullname}, with a musician of type {self.musician} and, an mkv order of')
            print(f' {self.mkvorder}, with a full path of {self.fullsongpath}.')

        elif self.musician in ('Mozart', 'mozart', 'MOZART'):
            print(f'generating song with mkv order {self.mkvorder}, and and a musician of type {self.musician}')
            self.corpusfiles = (MidiFile(f'training_music/single_track/Frédéric Chopin/{file_name}') for file_name in
                                listdir('training_music/single_track/Frédéric Chopin'))
            print('built corpus files')

            self.messages = (message for f in self.corpusfiles for message in
                             ['START'] + f.tracks[len(f.tracks) - 1] + ['END'])
            print('built messages')

            mkv = MarkovModel(midi_data=self.messages, order=self.mkvorder)
            print('Built Markov')

            gen_midi = MidiFile(type=1)
            track = MidiTrack()
            gen_midi.tracks.append(track)

            for self.message in mkv.sample():
                track.append(self.message)
                print(f'appended {self.message} to track')

            gen_midi.save(self.fullsongpath)
            print(
                f'created and, saved {self.songfullname}, with a musician of type {self.musician} and, an mkv order of')
            print(f' {self.mkvorder}, with a full path of {self.fullsongpath}.')

        elif self.musician in ('Beethoven', 'Beethoven', 'BEETHOVEN'):
            print(f'generating song with mkv order {self.mkvorder}, and and a musician of type {self.musician}')
            self.corpusfiles = (MidiFile(f'training_music/single_track/Ludwig van Beethoven/{file_name}') for file_name
                                in
                                listdir('training_music/single_track/Ludwig van Beethoven'))
            print('built corpus files')

            self.messages = (message for f in self.corpusfiles for message in
                             ['START'] + f.tracks[len(f.tracks) - 1] + ['END'])
            print('built messages')

            mkv = MarkovModel(midi_data=self.messages, order=self.mkvorder)
            print('Built Markov')

            gen_midi = MidiFile(type=1)
            track = MidiTrack()
            gen_midi.tracks.append(track)

            for self.message in mkv.sample():
                track.append(self.message)
                print(f'appended {self.message} to track')

            gen_midi.save(self.fullsongpath)
            print(
                f'created and, saved {self.songfullname}, with a musician of type {self.musician} and, an mkv order of')
            print(f' {self.mkvorder}, with a full path of {self.fullsongpath}.')

        else:
            print(f'generating song with mkv order {self.mkvorder}, and and a musician of type {self.musician}')
            self.corpusfiles = (MidiFile(f'training_music/single_track/Frédéric Chopin/{file_name}') for file_name in
                                listdir('training_music/single_track/Frédéric Chopin'))
            print('built corpus files')

            self.messages = (message for f in self.corpusfiles for message in
                             ['START'] + f.tracks[len(f.tracks) - 1] + ['END'])
            print('built messages')

            mkv = MarkovModel(midi_data=self.messages, order=self.mkvorder)
            print('Built Markov')

            gen_midi = MidiFile(type=1)
            track = MidiTrack()
            gen_midi.tracks.append(track)

            for self.message in mkv.sample():
                track.append(self.message)
                print(f'appended {self.message} to track')

            gen_midi.save(self.fullsongpath)
            print(
                f'created and, saved {self.songfullname}, with a musician of type {self.musician} and, an mkv order of')
            print(f' {self.mkvorder}, with a full path of {self.fullsongpath}.')


class SongGenBuilder():
    f"""
    This class contains the necessary functions to create a serialized markov model.
    That markov model can then be used to generate music. However right now the files
    that this class creates with the create model function are MASSIVE the one that I
    created while I was testing was almost 1.7GB in size. Because of this I do not recommend
    using this class unless you are creating a final model for some sort of production, or until
    I am able to shrink the size of the serialized markov. (The arguments of this class are defined in the __init__ function
    of this class.)

    """

    def __init__(self, mkvorder, pklmodelname, outputdir, trainingdir):
        """
        This is just the __init__ function for the SongGenBuilder class.

        :param mkvorder: (int) The order of the markov chain that gets serialized.
        :param pklmodelname: (str) The filename of the .pkl file that will contain the model.
        :param outputdir: (str) The directory for the produced file to be stored.
        :param trainingdir: (str) The directory that contains the midi file for the model to train on.
        """

        self.mkvorder = mkvorder
        self.pklmodelname = pklmodelname
        self.outputdir = outputdir
        self.trainingdir = trainingdir
        self.corpusfiles = None
        self.messages = None
        self.fullpathoutputpath = str(self.outputdir) + '/' + str(self.pklmodelname) + '.pkl'

    def get_mkv_order(self):
        """
        Gets the current 'self.mkvorder'.

        :return: self.mkvorder
        """
        return self.mkvorder

    def get_pickle_model_name(self):
        """
        Gets the current 'self.pklmodelname'.

        :return: self.pklmodelname
        """
        return self.pklmodelname

    def get_output_directory(self):
        """
        Gets the current 'self.outputdir'.

        :return: self.ouputdir
        """
        return self.outputdir

    def get_training_directory(self):
        """
        Gets the current 'self.trainingdir'.

        :return:  self.trainingdir
        """
        return self.trainingdir

    def set_mkv_order(self, mkvorder):
        """
        Sets self.mkvorder to desired value

        :param mkvorder: (int) The order of the markov chain that gets serialized.
        :return: self.mkvorder
        """
        self.mkvorder = mkvorder
        return self.mkvorder

    def set_pickle_model_name(self, pklmodelname):
        """
        Sets self.pklmodelname to desired value

        :param pklmodelname: The filename of the .pkl file that will contain the model.
        :return: self.pklmodelname
        """
        self.pklmodelname = pklmodelname
        return self.pklmodelname

    def set_output_directory(self, outputdir):
        """
        Sets self.outputdir to desired value.

        :param outputdir: The directory for the produced file to be stored.
        :return: self.output dir
        """
        self.outputdir = outputdir
        return self.outputdir

    def set_training_directory(self, trainingdir):
        """
        Sets self.trainingdir to desired value

        :param trainingdir: The directory that contains the midi file for the model to train on.
        :return: self.trainingdir
        """
        self.trainingdir = trainingdir
        return self.trainingdir

    def create_model(self):
        """
        I would say that this method contains the main functionality of the class.
        It actually writes the markov chain to a file for later use

        :return: None
        """
        print(f'generating model with an mkv order of {self.mkvorder}, and a training relative training dir of')
        print(f' {self.trainingdir}')
        self.corpusfiles = (MidiFile(f'{self.trainingdir}/{file_name}') for file_name in
                            listdir(f'{self.trainingdir}'))
        print('built corpus files')

        self.messages = (message for f in self.corpusfiles for message in
                         ['START'] + f.tracks[len(f.tracks) - 1] + ['END'])
        print('built messages')

        mkv = MarkovModel(midi_data=self.messages, order=self.mkvorder)
        print('Built Markov')

        with open(f'{self.fullpathoutputpath}', 'wb') as f:
            pickle.dump(obj=mkv, file=f)
            f.close()

class DirNumFiles():

    def __init__(self, directory):
        self.directory = directory
        self.numfileindir = len(listdir(self.directory))

    def get_directory(self):
        return self.directory

    def get_number_of_files_in_directory(self):
        return self.numfileindir

    def set_directory(self, directory):
        self.directory = directory
        return self.directory

class MidiGenerator1():

    def __init__(self, mkvorder, songname, songnumber, outputdir, mkvorderadditive=0,
                 lengthadd=1, lengthnegate=1, starterlength=500):
        self.mkvorder = mkvorder
        self.songname = songname
        self.songnumber = songnumber
        self.outputdir = outputdir
        self.fullsongname = str(self.songname) + str(self.songnumber) + '.mid'
        self.fullpath = str(self.outputdir) + '/' + str(self.fullsongname)
        self.mkvorderadditive = mkvorderadditive
        self.lengthadd = lengthadd
        self.lengthnegate = lengthnegate
        self.trainingdirectories = []
        self.numfilesintrainingdirectories = []
        self.corpusfiles = None
        self.messages = None
        self.starterlength = starterlength

    def get_markov_order(self):
        return self.mkvorder

    def get_song_name(self):
        return self.songname

    def get_song_number(self):
        return self.songnumber

    def get_output_directory(self):
        return self.outputdir

    def set_markov_order(self, mkvorder):
        self.mkvorder = mkvorder
        return self.mkvorder

    def set_song_name(self, songname):
        self.songname = songname
        return self.songname

    def set_song_number(self, songnumber):
        self.songnumber = songnumber
        return self.songnumber

    def set_output_directory(self, outputdir):
        self.outputdir = outputdir
        return self.outputdir

    def add_training_directory(self, trainingdirfp):

        if trainingdirfp not in self.trainingdirectories:
            self.trainingdirectories.append(trainingdirfp)
        else:
            print(f'Sorry {trainingdirfp} is already a training directory')
        return self.trainingdirectories

    def delete_training_directory(self, trainingdirfp):
        self.trainingdirectories.remove(trainingdirfp)
        return self.trainingdirectories

    def generate_midi(self, equalizetrainingdirs):

        print()

        if len(self.trainingdirectories) > 0:
            if equalizetrainingdirs == False:

                for trainingdirectory in self.trainingdirectories:
                    self.corpusfiles = (MidiFile(f'{trainingdirectory}/{file_name}') for file_name in
                                        listdir(f'{trainingdirectory}'))
                    print(f'Built corpus files for training directory: {trainingdirectory}')

                    self.messages = (message for f in self.corpusfiles for message in
                                     ['START'] + f.tracks[len(f.tracks) - 1] + ['END'])
                    print(f'Built messages for training directory: {trainingdirectory}')
                    print("")
            if equalizetrainingdirs == True:

                for trainingdirectory in self.trainingdirectories:
                    num_files_in_dir = len(listdir(f'{trainingdirectory}'))
                    print(num_files_in_dir)
                    self.numfilesintrainingdirectories.append(num_files_in_dir)
                    self.numfilesintrainingdirectories.sort()
                    print(self.numfilesintrainingdirectories)
                print()
                lowest_number_of_files_in_directory = min(self.numfilesintrainingdirectories)
                print(lowest_number_of_files_in_directory)
                print()

                for trainingdirectory in self.trainingdirectories:
                    list_of_files = listdir(f'{trainingdirectory}')
                    number_of_file_in_dir = len(listdir(f'{trainingdirectory}'))
                    subtract_num = number_of_file_in_dir - lowest_number_of_files_in_directory
                    print(number_of_file_in_dir)

                    for i in range(0, subtract_num):
                        random_file = random.choice(list_of_files)
                        list_of_files.remove(random_file)

                    self.corpusfiles = (MidiFile(f'{trainingdirectory}/{file_name}') for file_name in
                                        list_of_files)
                    print(f'Built corpus files for training directory: {trainingdirectory}')

                    self.messages = (message for f in self.corpusfiles for message in
                                     ['START'] + f.tracks[len(f.tracks) - 1] + ['END'])
                    print(f'Built messages for training directory: {trainingdirectory}')
                    print(f'the directory with the lowest files file number is {lowest_number_of_files_in_directory}')
                    print(len(list_of_files))
                    print()

            mkv = MarkovModel(midi_data=self.messages, order=self.mkvorder, orderadditive=self.mkvorderadditive,
                              lengthadditive=self.lengthadd, lengthnegate=self.lengthnegate, starterlength=self.starterlength)

            gen_midi = MidiFile(type=1)
            track = MidiTrack()
            gen_midi.tracks.append(track)

            for self.message in mkv.sample():
                track.append(self.message)
                print(f'appended {self.message} to track')

            midi_length = gen_midi.length
            gen_midi.save(self.fullpath)
            print(f'created and, saved {self.fullsongname},and a markov order order of')
            print(f' {self.mkvorder}, a length of {midi_length} with a full path of {self.fullpath}.')

        else:
            print("Before you can generate a midi file you need to specify at least 1 directory that holds midi's")
            print(" train on. To give a training directory use the .add_training_directory method of this class")


def get_average_length_of_midi_files_in_directory(directory):
    files_in_dir = listdir(directory)
    midi_lengths_in_dir = []

    for file in files_in_dir:
        path = f'{directory}/{file}'
        file_path = Path(path)

        #if file_path.suffix not in ('mid', '.mid', '.midi', 'midi'):
            #files_in_dir.remove(file)

        #else:
            #midi_file = MidiFile(file)
            #midi_file_length = midi_file.length
            #midi_lengths_in_dir.append(midi_file_length)

        midi_file = MidiFile(path)
        midi_file_length = midi_file.length
        midi_lengths_in_dir.append(midi_file_length)

    sum_of_midi_lengths_in_dir = sum(midi_lengths_in_dir)
    length_of_midi_lengths_in_dir = len(midi_lengths_in_dir)
    average_midi_length_of_dir = sum_of_midi_lengths_in_dir / length_of_midi_lengths_in_dir

    return average_midi_length_of_dir