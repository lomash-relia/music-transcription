# Music Transcription Model

## Overview
This repository contains a simple Convolutional Neural Network (CNN) model that is designed to identify 37 different guitar notes individually.

## Features
- **Guitar Note Recognition:** The model is capable of identifying 37 distinct guitar notes, enabling accurate transcription of guitar music.
- **Simplicity:** The model is kept simple for ease of understanding and modification, making it suitable for beginner developers.

## Getting Started
1. **Clone the Repository:**
   ```
   git clone https://github.com/lomash-relia/music-transcription-model.git
   ```

2. **Install Dependencies:**
   ```
   pip3 install torch torchaudio torchsummary scikit-learn
   ```

3. **Run the Model:**
   ```
   python inference.py
   ```

## Model Training
If you wish to train the model with your own dataset, follow these steps:

1. **Prepare Dataset:**
   Organize your guitar note dataset with labeled examples for each of the 37 notes.

2. **Configure Training Settings:**
   Adjust the hyperparameters and training settings in the `train.py` file according to your dataset and preferences.

3. **Train the Model:**
   ```
   python train.py
   ```
   This will initiate the training process.

## Contribution Guidelines
Contributions are welcome! If you have suggestions, enhancements, or bug fixes, please follow these guidelines:

- Fork the repository.
- Create a new branch for your changes.
- Make your changes and test thoroughly.
- Create a pull request, explaining the changes made and providing any necessary context.

Feel free to explore, use, and contribute to enhance the functionality of this music transcription model. Happy coding!