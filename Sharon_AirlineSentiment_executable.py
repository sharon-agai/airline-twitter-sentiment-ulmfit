# -*- coding: utf-8 -*-
"""
ULMfit fine-tuned to Twitter sentiment data, and a classifier used to
predict the sentiment of other tweets
    INPUT: a csv that includes tweets and airline sentiment (among other things,
    but those are key)
    OUTPUT: a confusion matrix describing the model's performance on the validation
    set
"""
from fastai import *
from fastai.text import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from torchtext import vocab

sns.set(style="whitegrid")

raw_tweets = pd.read_csv('Tweets.csv', encoding='latin-1')

# Cleaning and preprocessing
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

re1 = re.compile(r'  +')
def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

labelencoder = LabelEncoder()

raw_tweets['tidy_tweet'] = np.vectorize(remove_pattern)(raw_tweets['text'], "@[\w]*")
raw_tweets['tidy_tweet'] = raw_tweets['tidy_tweet'].str.encode('ascii', 'ignore').str.decode('utf-8', 'ignore')
raw_tweets['tidy_tweet'] = raw_tweets['tidy_tweet'].apply(fixup).values
raw_tweets['airline_sentiment'] = labelencoder.fit_transform(raw_tweets['airline_sentiment'])

# Creating data bunches
texts = raw_tweets['tidy_tweet']
sents = raw_tweets['airline_sentiment']
tweets_df = pd.concat([sents,texts], axis=1)
tweets_df.head()

data_lm = (TextList.from_df(tweets_df, cols='tidy_tweet')
            .random_split_by_pct(valid_pct=0.1)
            .label_for_lm()
            .databunch(bs=48))

data_clas = (TextList.from_df(tweets_df, cols='tidy_tweet', vocab=data_lm.vocab)
                   .random_split_by_pct(valid_pct=0.1)
                   .label_from_df(cols='airline_sentiment')
                   .databunch(bs=48))

# Initializing the model
learn = language_model_learner(data_lm, arch=AWD_LSTM, drop_mult=0.7, callback_fns=ShowGraph)

labelcounts = tweets_df.groupby(['airline_sentiment']).size()
label_sum = len(tweets_df['airline_sentiment'])
weights_balance = [(1-count/label_sum) for count in labelcounts]
loss_weights = torch.FloatTensor(weights_balance).cuda()
learn.crit = partial(F.cross_entropy, weight=loss_weights)

# Finetuning via gradual unfreezing and discriminative layer tuning
learn.unfreeze()
learn.fit_one_cycle(2, slice(5e-2/(2.6**4), 5e-2), moms = (0.8,0.7), wd=1e-2)
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(3e-2/(2.6**4), 3e-2), moms = (0.8,0.7), wd=1e-2)
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(3e-2/(2.6**4), 3e-2), moms = (0.8,0.7), wd=1e-2)
learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-2/(2.6**4), 2e-2), moms = (0.8,0.7), wd=1e-2)

# Save the tuned language model
learn.freeze()
learn.save('fine_tuned_lm')
learn.save_encoder('fine_tuned_lm_enc')

# Create a classifier using the tuned language model
learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.7, callback_fns=ShowGraph)
learn.load_encoder('fine_tuned_lm_enc')
learn.clip=0.3

labelcounts = tweets_df.groupby(['airline_sentiment']).size()
label_sum = len(tweets_df['airline_sentiment'])
weights_balance = [(1-count/label_sum) for count in labelcounts]
loss_weights = torch.FloatTensor(weights_balance).cuda()
learn.crit = partial(F.cross_entropy, weight=loss_weights)

# Finetuning the classifier
learn.unfreeze()
learn.fit_one_cycle(7, slice(2.5e-2/(2.6**4), 2.5e-2), moms = (0.8,0.7), wd=1e-2)
learn.freeze_to(-2)
learn.fit_one_cycle(2, slice(2e-2/(2.6**4), 2e-2), moms = (0.8,0.7), wd=1e-2)
learn.freeze_to(-3)
learn.fit_one_cycle(2, slice(2e-2/(2.6**4), 2e-2), moms = (0.8,0.7), wd=1e-2)
learn.unfreeze()
learn.fit_one_cycle(1, slice(1.5e-2/(2.6**4), 1.5e-2), moms = (0.8,0.7), wd=1e-2)

# Saving the classifier
learn.save_encoder('fine_tuned_clas')

# Get predictions, output a confusion matrix
learn.load_encoder('fine_tuned_clas')
y_pred, y_true = learn.get_preds(ds_type=DatasetType.Valid)
y_pred = np.array(np.argmax(y_pred, axis=1))
y_true = np.array(y_true)
cm = confusion_matrix(y_true, y_pred)
print(cm)
