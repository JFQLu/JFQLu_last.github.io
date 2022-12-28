## Tweet Sentiment Analysis

### Project Details
In the fast-paced world of social media, it is critical to accurately identify the sentiment expressed in online communication, as it can have a significant impact on a company's brand and profitability. With the constant flow of tweets, it can be difficult to determine whether a particular message will go viral in a positive way or have a negative impact on the brand. Capturing and understanding the sentiment conveyed through language is crucial for making timely and informed decisions. However, accurately identifying the specific words that contribute to the overall sentiment can be challenging.

The Kaggle competition [Tweet Sentiment Extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview) embodies the fundamental ideas of sentiment analysis. This challenge requires the participants to look at the labelled sentiment for a given tweet and determine the word or phrase that best supports it. I will be using this data for this project.

*For this project, I will be focussing on the classification of the sentiment for each tweet and optimisation of the implementation. I believe that this deliverable would be more useful for a company looking to do sentiment analysis on their own customers.* 

### Data
The raw data contains the columns textID, text, selected_text, and sentiment.
![image](https://user-images.githubusercontent.com/98208084/209868862-3cdb9179-cb41-4bfe-a284-d7d8476160fe.png)

We can remove the selected_text column since the sentiment analysis will be performed on the full text. Moreover, we assign dummy variables to the sentiment:
- positive &rarr; 1 
- neutral &rarr; 0 
- negative &rarr; -1

```python
df_train = df_train.drop(['textID', 'selected_text'], axis=1)
df_train['sentiment'] = df_train['sentiment'].apply(lambda x: 1 if x == 'positive' else 0 if x == 'neutral' else -1)

df_test = df_test.drop(['textID'], axis=1)
df_test['sentiment'] = df_test['sentiment'].apply(lambda x: 1 if x == 'positive' else 0 if x == 'neutral' else -1)
```

### Exploratory Data Anlysis
First we look at the label distribution in the training data. 

```python 
label = df_train['sentiment'].value_counts()
df_label = pd.DataFrame({'label_name':label.index[0:],'fre':label.values[0:]})

plt.pie(df_label.fre,labels=df_label.label_name,autopct='%1.2f%%')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/98208084/209869945-7005cbe4-d627-40f9-adaa-8fa56f7678bb.png)






