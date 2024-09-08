# Sentiment Analysis using Quantum Natural Language Processing (Q-NLP)

## Team Members
Naveen Nidadavolu (22MIA1049)
Shivam Thakur (22BPS1014)
Ram Sasidhar Putcha (22MIA1055)
Yasir Ahmad (22MIA1064)

## Project Overview
This project applies Quantum Machine Learning techniques to perform sentiment analysis on a dataset of tweets. By leveraging the power of quantum computation and natural language processing, we explore how quantum neural networks can enhance text classification tasks.

## Dataset
We used the Emotion Prediction with Quantum5 Neural Network AI dataset from Kaggle, which contains tweets labeled with different emotions. The dataset can be found here.

## Methodology
### 1. Data Preprocessing
Tokenization, stemming, and lemmatization of text data.
Removal of stop words and non-alphanumeric characters.
Padding sequences to ensure consistent input size.
### 2. Model Architecture
The model consists of both classical and quantum layers:
Quantum Layer: Encodes input data using quantum gates and applies parameterized quantum circuits.
Classical Layers: Dense layers connected sequentially to form a neural network for sentiment classification.
Optimizer: Stochastic Gradient Descent (SGD).
Loss Function: Mean Absolute Error (MAE).
### 3. Quantum Processing
We utilized the AngleEmbedding and StronglyEntanglingLayers templates from PennyLane to process the data quantum mechanically.
The quantum layer is implemented using a QNode that calculates expectation values of Pauli-Z operators.

## Dependencies
To run this project, you need to install the following libraries:

TensorFlow 2.7.0
TensorFlow Quantum 0.7.2
PennyLane
tf_keras (Legacy Keras)
Cirq 0.13.1
nltk
Installation
Install the required libraries:

Modify environment variables to ensure compatibility with legacy Keras:
lua

## Results
The model was trained for 5 epochs with a batch size of 256. The final prediction is obtained by evaluating the model on new input data.

## Future Work
Optimizing the quantum layer with more qubits and complex quantum gates.
Experimenting with different datasets for broader application.

## Citation
If you use this project or dataset, please cite the following:
Emirhan BULUT. (2022). Emotion Prediction with Quantum5 Neural Network AI [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/2129637
