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

We observe that there is 10% more neural sentiment tweets compared to the negative and positive classes which may be a potential issue biasing models toward the more prevalent class. Since the difference is only 10% we will ignore this however a potential remedy for this would be to oversample the less prevelent classes or to undersample the more prevelent class.

Now, observing tweet length distibutions across sentiments, 

```python
plt.figure(figsize=(16, 4))
plt.subplot(1, 3, 1)
df_train.query("sentiment==-1")["text"].str.len().plot(kind="hist", title="Negative")
plt.xlabel('Tweets Length')

plt.subplot(1, 3, 2)
df_train.query("sentiment==0")["text"].str.len().plot(kind="hist", title="Neutral")
plt.xlabel('Tweets Length')

plt.subplot(1, 3, 3)
df_train.query("sentiment==1")["text"].str.len().plot(kind="hist", title="Positive")
plt.xlabel('Tweets Length')

plt.show()
```

![image](https://user-images.githubusercontent.com/98208084/209871294-bc12eba4-1ca7-40e9-9b71-a1905ea723b2.png)

and word count distributions,

```python
plt.figure(figsize=(16, 4))
plt.subplot(1, 3, 1)
df_train.query("sentiment==-1").text.map(lambda x: len(x.split())).plot(kind="hist", title="Negative")
plt.xlabel('Number of Words')

plt.subplot(1, 3, 2)
df_train.query("sentiment==0").text.map(lambda x: len(x.split())).plot(kind="hist", title="Neutral")
plt.xlabel('Number of Words')

plt.subplot(1, 3, 3)
df_train.query("sentiment==1").text.map(lambda x: len(x.split())).plot(kind="hist", title="Positive")
plt.xlabel('Number of Words')

plt.show()
```
![image](https://user-images.githubusercontent.com/98208084/209871517-c977b35a-0025-4e02-b5ad-56c920696a94.png)

We see from this that there is little difference in terms of number of characters or words across each sentiment.

### Preprocessing
#### Decontraction
Decontraction is the process of converting contractions, which are shortened forms of words or phrases, back to their full form. For example, "I'm" would be converted to "I am". 

Decontraction is important in natural language processing (NLP) because it helps to normalize the text and make it easier for NLP models to process and understand. Contractions can be ambiguous and can have multiple meanings depending on the context in which they are used. We perform decontraction manually using the following code.

```python
def decontraction_text(text):    
    # performing de-contraction
    text = text.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'").replace("`", "'")\
                .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                .replace("he's", "he is").replace("she's", "she is").replace("'s", " is")\
                .replace("'m", " am").replace("'d", " would")\
                .replace("'ll", " will")
    return text
    
df_train['text']= df_train['text'].apply(lambda x : decontraction_text(x))
df_test['text'] = df_test['text'].apply(lambda x: decontraction_text(x))
df_train.head()
```
Another important step in preprocessing is stemming. Stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form. For example, the stem of the word "stemmer," is "stem."

Stemming is important in natural language processing (NLP) because it helps to reduce the dimensionality of the data and make it easier for NLP models to process and understand. It does this by reducing the number of unique word forms that need to be considered









