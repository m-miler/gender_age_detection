# Gender and Age Detection with Tensorflows and OpenCV

# Introduction

In this project I would like to create a simple tkinter application, which will allow users to recognize gender and age. 
Application has two options: 

1. Use a web camera to regognize face and predict gender and age in real time 
2. Load a sample image from a directory than predict gender and age 

![alt text](https://github.com/m-miler/gender_age_detection/blob/master/results/application_result.PNG)

# About Project

We will use the deep learning methods to train a convolutiona neural network that will predict age groups and gender from an image containing the face of a person.


# Dataset

The dataset contains verious images in various real-world conditions with diffrent lighting and noise level. 
It can be downloaded in zip format (links below). The directory contains the following files:

Statistic

* Total number of photos: 26,580
* Total number of subjects: 2,284
* Number of age groups / labels: 8 (0-2, 4-6, 8-13, 15-20, 25-32, 38-43, 48-53, 60+)
* Age labels: Yes
* In the wild: Yes
* Subject labels: Yes

Links:
1. https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification
2. https://talhassner.github.io/home/projects/Adience/Adience-data.html

# Results

1. Real Tiem Result

![alt text](https://github.com/m-miler/gender_age_detection/blob/master/results/real_time_result.PNG)

2. Load Images Results

![alt text](https://github.com/m-miler/gender_age_detection/blob/master/results/leonard_result.png)
![alt text](https://github.com/m-miler/gender_age_detection/blob/master/results/penny_result.PNG)
![alt text](https://github.com/m-miler/gender_age_detection/blob/master/results/freinds.PNG)
![alt text](https://github.com/m-miler/gender_age_detection/blob/master/results/baby_result.PNG)


# Next steps

1. Improve age model accuaracy(find more examples and use data augmentation) 
2. Reduce overfiting in both models
