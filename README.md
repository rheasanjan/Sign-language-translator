# ISL-Translator


While there is evidence to suggest that sign language was developed as early as the 5th century BC for the benefit of deaf people, it is only now that attempts are being made to automate the translation of sign language to oral language. This project is one such attempt, and aims at building a  system which recognizes Indian Sign Language. The gestures are captured by a webcam and the images are subjected to image processing techniques and feature extraction for the identification of the gestures. The features are then fed as input to a Support Vector Machine, which classifies the gestures and converts them to corresponding text.

### Installation

This requires you to have any version of python higher than 3.0. Run the following command to install the dependencies.

```sh
pip install -r requirements.txt
```

After you have installed the dependencies, run the following command

```sh
python ISL1.py
```

To check the accuracy, run the following command

```sh
python testing_accuracy.py
```
