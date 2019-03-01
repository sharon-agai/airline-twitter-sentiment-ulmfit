# Twitter Sentiment Analysis using ULMfit
## By Sharon Agai

In this project, I fit a ULMfit model to labeled Twitter data and use a classifier on top of the finetuned language model
to classify the sentiment of tweets. 

The original dataset can be found here, on Kaggle: <br>
https://www.kaggle.com/crowdflower/twitter-airline-sentiment#Tweets.csv

### This repository contains: 
- [Sharon_AirlineSentiment_executable.py](../blob/master/Sharon_AirlineSentiment_executable.py) <br>
An executable file that takes in a csv file of tweets, runs my model, and outputs a confusion matrix with the model's performance

- [Sharon_AirlineSentiment_explained.ipynb](../blob/master/Sharon_AirlineSentiment_explained.ipynb) <br>
A Google Colab notebook that includes data visualization, explains my finetuning process for ULMfit and the classifier, and includes the discussion and interpretation of my results. 

- [Tweets.csv](../blob/master/Tweets.csv) <br> A copy of the dataset I used
