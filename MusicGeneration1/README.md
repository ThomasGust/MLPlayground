# Classical Piano Music Generator

This application generates piano music in midi form using an nth order Markov model trained off of classical piano music.
A Flask web app is included as a simple interface to generate musical pieces using Markov models of order 1 through 6. 

Composers currently in corpus data: **Chopin**

## Getting Started

These instructions will get you a copy of this project up and running on your local machine so you can 
generate your own music and add your own midi files to the training data.

### Prerequisites

This is a Python 3 project and requires the latest version of Python and a virtual environment. 

### Installing

1. Download or clone this repository.

2. Set up your python virtual environment in the cloned directory.

3. Install all required packages from requirements.txt.

### Adding midi files to training_music

1. Add multi-track midi files to *training_music/multi_track*.

2. Run `python merge_midi_tracks.py training_music/multi_track training_music/single_track` to merge all midi files from *training_music/multi_track* from multi track midi files to single track midi files and save them in *training_music/single_track*. It is necessary for midi files to be single track for them to be used in training the Markov model. 

This is Thomas now, at this point I have made drastic modifications to the code, and you should just be able to add data to
the training directory directly, and the generator should be able to take in multiple tracks.

### Running the code

To generate a midi file from the training music, run `python generate_midi.py <order>` where order is the order of the markov model that should be used to generate the music. The resulting midi file is stored in *static/generated.mid*.

To run the Flask app, run `python app.py` and visit your localhost server. 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgements

* Training music from [www.piano-midi.de](http://www.piano-midi.de/)
* merge_midi_tracks.py from [Star Brilliant](https://github.com/m13253)

