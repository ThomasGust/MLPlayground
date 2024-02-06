from MusicGeneration1.utils import MidiGenerator1, get_average_length_of_midi_files_in_directory
import math

SongGen = MidiGenerator1(mkvorder=9999, songname='MultiGeneratedTest', songnumber=47,
                         outputdir='generated/Testing', mkvorderadditive=0, lengthadd=3, lengthnegate=3,
                         starterlength=10000)

SongGen.add_training_directory(trainingdirfp='training_music/single_track/Ludwig van Beethoven')
SongGen.add_training_directory(trainingdirfp='training_music/single_track/Wolfgang Amadeus Mozart')
SongGen.add_training_directory(trainingdirfp='training_music/single_track/Frédéric Chopin')

SongGen.generate_midi(equalizetrainingdirs=True)
print()
directory = str(SongGen.get_output_directory())
average_length_of_midi_files_in_directory = get_average_length_of_midi_files_in_directory(directory=directory)
print(f"directory {directory}'s midi files have an average length of {average_length_of_midi_files_in_directory}")
print(f' in seconds, or about {math.ceil(average_length_of_midi_files_in_directory / 60)} minutes')