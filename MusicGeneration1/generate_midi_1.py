from MusicGeneration1.utils import generate_random_song_key, MidiGenerator

song_key = generate_random_song_key()
Song_generator = MidiGenerator(mkvorder=600, songname='generatedTestingwithMKVOrder600', songnumber=1, musician='Beethoven'
              , outputdir='generated/Testing')

Song_generator.generate_midi()