# Facial Emotion Recognition
<p align="center">
  <img width="900" height="400" src="https://miro.medium.com/max/5000/1*Z2p6ohPSv2vdM-JzYYRCng.png">
</p>

Facial emotion recognition is the process of detecting human emotions from facial expressions. The human brain recognizes emotions automatically, and software has now been developed that can recognize emotions as well. This technology is becoming more accurate all the time, and will eventually be able to read emotions as well as our brains do. 

AI can detect emotions by learning what each facial expression means and applying that knowledge to the new information presented to it. Emotional artificial intelligence, or emotion AI, is a technology that is capable of reading, imitating, interpreting, and responding to human facial expressions and emotions.

Humans are used to taking in non verbal cues from facial emotions. Now computers are also getting better to reading emotions. So how do we detect emotions in an image? We have used an open source data set — Face Emotion Recognition (FER) from Kaggle and built a CNN to detect emotions. The emotions can be classified into 7 classes — happy, sad, fear, disgust, angry, neutral and surprise.

The project is implemented in Python using Tensorflow along with Keras.

_________________


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/)

```bash
pip install opencv-python
pip install tensorflow
pip install keras
```

## Implementation

### Pre-requisites

* You should have an understanding of convolutional neural networks. 

* Basic OpenCV usage.

* Building Neural Nets in Keras.


### 1. Datasets
* Challenges in Representation Learning: Facial Expression Recognition Challenge: [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)


### 2. Project Files Structure
```.
├── Data Analysis and Model Building.ipynb
├── emotion.py
├── fer2013
│   ├── fer2013.bib
│   ├── fer2013.csv
│   └── README
├── FER_model.h5
└── haarcascade_frontalface_default.xml

```

1. ```Data Analysis and Model Building.ipynb : ``` This notebook contains the code for Data-Preprocessing and Model Building.
2. ```fer2013.csv : ``` This file contains the dataset.


3. ```FER_model.h5: ``` This is the model checkpoint on completion of training.

4. ```haarcascade_frontalface_default.xml : ``` Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones in their paper, "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.You can find out more about it from [here](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html).

5. ```emotion.py : ``` This file includes the complete pipeline from reading images from your webcam using OpenCV to detecting the Facial Emotion using the trained Keras Model.


## Usage

```python emotion.py```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements

Would like to thank [Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) for providing a good dataset.
