# Application of Machine Learning Tools for Music Composition

Music composition using machine learning has developed a lot in the last few years. Music is an art that brings out the emotions, expressions, and creativity of an artist. Particularly, songs composed in the 1970s have a unique style and were totally dependent on the skills of an artist. Machine learning algorithms are able to capture the structure of music with ease. In this project, we explore various machine learning algorithms that are able to generate music similar to the songs composed by famous American guitarist, Nile Rodgers. Three different models have been used to generate music. First, we attempt to generate music using the first-order Markov chain, as a music piece is seen as a sequence of states with each state being the chords of the guitar. The results from Markov chain are random as it is conditioned only on one previous event. Second, we compose music using the Long Short Term Memory network that is designed to work efficiently for long temporal sequences. The results from LSTM are better sounding than the random sounds generated by Markov Chain. Third, a 1D Convolutional Neural Network is implemented. The training time improves drastically from the LSTM.

## Getting Started

This repository contains the code for the thesis project. Supporting functions are defined in the custom.py file. 

* read_midi() reads chords from the input midi file
* read_data() reads all midi files present in the midi folder
* prepare_data() generates input data and targets for the deep learning models
* get_transition_matrix() generates a transition matrix for the input chords data
* generate_stochastic_music() generates music from the first-order Markov chain model
* pick_chord_probs() selects a chord based on the probability in the transition matrix
* generate_music() generates music from the two trained deep learning models
* convert_to_midi() generates midi file for the predicted chords
* cnn() function to define 1dcnn architecture
* lstm() function to define lstm architecture

The ml_for_music.ipynb file contains the body of the code. 

### Prerequisites

*  Music21
*  Keras

## Author

* **Idris Mustafa** (u6733671)
