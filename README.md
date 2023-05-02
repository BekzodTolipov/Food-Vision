# Food-Vision

Food Vision is a deep learning project that aims to classify food images into different categories. The project uses the EfficientNetB0 architecture and the Food101 dataset. The model was trained and evaluated using TensorFlow and Keras. The main files in the project are:

`food_vision.ipynb`: A Jupyter notebook that contains the code for the project. It includes data preprocessing, model creation, training, and evaluation.
`app.py`: A Python script that can be used to make predictions on new images using a trained model.
`requirements.txt`: A file that lists the dependencies required to run the project.

The project achieved an accuracy of approximately 75.61% on the test set. The model is capable of recognizing a variety of food categories, including sushi, pizza, and steak.

The project can be improved by fine-tuning the model's hyperparameters, increasing the amount of training data, and using data augmentation techniques.

----

## Charts and Analysis

![confusion matrix](/images/confusion_matrix.png)

![f1-score 101 classes](/images/f-1score-101-classes.png)

![misclassification example](/images/miss_examples.png)

----

## Playground

You can interact with the model by visiting the following link: [app-link](https://bekzodtolipov-food-vision-app-y8dlog.streamlit.app/)