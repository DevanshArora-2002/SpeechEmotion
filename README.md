# SpeechEmotionRecognition
# Overview
The Speech Emotion Recognition (SER) Model is an advanced deep learning model designed to recognize and classify human emotions from speech data. It's built to cater to a wide range of applications such as mental health monitoring, human-robot interaction, customer sentiment analysis, and more.
# Dataset
RAVDESS is a multimodal database for the recognition of emotional states in speech and song. It includes 7356 files rated by 247 individuals 10 times on emotional validity, intensity, and genuineness. The database is gender balanced consisting of 24 professional actors, vocalizing lexically-matched statements in a neutral North American accent. Each statement is expressed in two forms: speech and song, across a range of emotions.

On the other hand, CREMA-D is a crowd-sourced dataset that consists of over 7500 original clips from 91 actors aged between 20 to 74 from various ethnic backgrounds, showcasing a variety of emotions. The actors were asked to vocalize the same phrases with different emotional expressions, providing a rich source of variance in both age and ethnicity. This dataset aids in building robust models that perform well across different demographics.
# Model Architecture
Our model's architecture is based on the Transformer, originally introduced in the paper "Attention is All You Need". Unlike traditional RNNs or CNNs, Transformers use a mechanism called self-attention that allows them to process input sequences without the need for explicit sequential computation. This is particularly useful for processing long sequences of audio data in speech emotion recognition.

The model takes in raw audio data, which is then converted into spectrograms to represent the features in a more structured form. These are then passed through several Transformer encoder layers, which produce an output sequence that represents the emotion content in the input audio. The output sequence is then passed through a fully connected layer to produce the final emotion classification.
# Python files
command line code
python3 filename.py --mode (train/test) --epochs --output_dir_name --dataset_path --model (Optional for model weights)

