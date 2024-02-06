from mido import Message, MidiFile, MidiTrack
from MusicGeneration1.model.markov_model import MarkovModel
from os import listdir
import sys
import random
import math
from MusicGeneration1.utils import MidiGenerator, generate_random_song_key, randomString, get_time

# I don't really know why I am keeping this file around. I created a much better version of it in the form of a class
# which is stored in utils.py, and the I have a new python file (generated_midi_1.py) that uses that classes .generatemidi()
# method to actually create the music.

# So if you want to generate music just run the new generation script

artist_prompt = input('Type Mozart or Beethoven to create a piece that sounds like theirs, (the defuault is Chopin):')
artist=None
if artist_prompt.lower() == 'mozart':
    artist = 'Mozart'
elif artist_prompt.lower() == 'beethoven':
    artist = 'Beethoven'
else:
    artist = 'Chopin'

def generate_midi(mkv_order=4, song_number=101, song_name='generated'):
    """
    Generates and saves a midi file using an Nth order markov chain trained on midi files from training_music/single_track.
    Generated midi file is saved in the generated directory.
    Args:
        mkv_order (int): The order of the markov chain that should be used to generate the file
    """
    if artist == 'Chopin':
        print(f'generating song with key {song_key}, and an artist of type {artist}')
        corpus_files = (MidiFile(f'training_music/single_track/Frédéric Chopin/{file_name}') for file_name in
                        listdir('training_music/single_track/Frédéric Chopin'))
        print('built corpus files')

        messages = (message for f in corpus_files for message in ['START'] + f.tracks[len(f.tracks) - 1] + ['END'])
        print('built messages')

        mkv = MarkovModel(midi_data=messages, order=mkv_order)
        print('Built Markov')

        # Type 0 single track file
        gen_midi = MidiFile(type=1)
        track = MidiTrack()
        gen_midi.tracks.append(track)

        for message in mkv.sample():
            track.append(message)
            print(f'appended {message} to track')

        song_full_name = f'{song_name}{song_number}'

        gen_midi.save(f'generated/Frédéric Chopin/{song_full_name}.mid')
        print(f'created and saved {song_full_name}.mid, with a song seed of {mkv_order}')
    elif artist == 'Mozart':
        print(f'generating song with key {song_key}, and an artist of type {artist}')
        corpus_files = (MidiFile(f'training_music/single_track/Wolfgang Amadeus Mozart/{file_name}') for file_name in
                        listdir('training_music/single_track/Wolfgang Amadeus Mozart'))
        print('built corpus files')

        messages = (message for f in corpus_files for message in ['START'] + f.tracks[len(f.tracks) - 1] + ['END'])
        print('built messages')

        mkv = MarkovModel(midi_data=messages, order=mkv_order)
        print('Built Markov')

        # Type 0 single track file
        gen_midi = MidiFile(type=1)
        track = MidiTrack()
        gen_midi.tracks.append(track)

        for message in mkv.sample():
            track.append(message)
            print(f'appended {message} to track')

        song_full_name = f'{song_name}{song_number}'

        gen_midi.save(f'generated/Wolfgang Amadeus Mozart/{song_full_name}.mid')
        print(f'created and saved {song_full_name}.mid, with a song seed of {mkv_order}')
    elif artist == 'Beethoven':
        print(f'generating song with key {song_key}, and an artist of type {artist}')
        corpus_files = (MidiFile(f'training_music/single_track/Ludwig van Beethoven/{file_name}') for file_name in
                        listdir('training_music/single_track/Ludwig van Beethoven'))
        print('built corpus files')

        messages = (message for f in corpus_files for message in ['START'] + f.tracks[len(f.tracks) - 1] + ['END'])
        print('built messages')

        mkv = MarkovModel(midi_data=messages, order=mkv_order)
        print('Built Markov')

        # Type 0 single track file
        gen_midi = MidiFile(type=1)
        track = MidiTrack()
        gen_midi.tracks.append(track)

        for message in mkv.sample():
            track.append(message)
            print(f'appended {message} to track')

        song_full_name = f'{song_name}{song_number}'

        gen_midi.save(f'generated/Ludwig van Beethoven/{song_full_name}.mid')
        print(f'created and saved {song_full_name}.mid, with a song seed of {mkv_order}')

song_key = generate_random_song_key()
generate_midi(mkv_order=song_key, song_number=12, song_name=f'generated{artist}')