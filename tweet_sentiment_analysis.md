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

#### Lemmatization
Lemmatization is another useful preprocessing technique for NLP. Lemmatization is the process of reducing a word to its base form, or lemma. For example, the lemma of the word "was" is "be," and the lemma of the word "better" is "good."

Lemmatization is important in natural language processing (NLP) because it helps to reduce the dimensionality of the data and make it easier for NLP models to process and understand. It does this by reducing the number of unique word forms that need to be considered, which can make it easier to identify common themes and patterns in the text. The code to do this is below using the nltk implementation. 

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemma_stem(x):
    new = ""
    for y in x.split():
        lemmatizer.lemmatize(y)
        new = new + y + " "
    return new.rstrip()
    
df_train['text'] = df_train['text'].apply(lambda x:lemma_stem(x))
df_test['text'] = df_test['text'].apply(lambda x: lemma_stem(x))
```

#### Removing URLs
Next we can remove URLs as they do not provide useful information,
```python 
def remove_URLs(text):
    return re.sub(r'http\S+', ' ', text,  flags=re.MULTILINE)
df_train['text']= df_train['text'].apply(lambda x : remove_URLs(x))
``` 

#### Removing Punctuations
Whether or not to remove punctuations in natural language processing (NLP) tasks depends on the specific task and the type of punctuation being used. In some cases, punctuations can be useful for understanding the meaning of the text and should be retained. In other cases, punctuations may not be necessary and can be removed as a preprocessing step. 

In our case I choose to remove punctuations to reduce the complexity of the final models and given the context of short tweets, punctuations may not be very useful. 

```python 
def lower_text_and_remove_special_chars(text):
    text = text.lower().strip()
    return re.sub(r"\W+", " ", text)

df_train['text']= df_train['text'].apply(lambda x : lower_text_and_remove_special_chars(x))
df_test['text'] = df_test['text'].apply(lambda x: lower_text_and_remove_special_chars(x))
```

#### Removing HTML tags
HTML tags should be removed as they do not provide information on the sentiment of a tweet.

```python 
from bs4 import BeautifulSoup
def remove_html_tags(text):
    return BeautifulSoup(text).get_text()

df_train['text']= df_train['text'].apply(lambda x : remove_html_tags(x))
df_test['text'] = df_test['text'].apply(lambda x: remove_html_tags(x))
```

#### Spelling correction
There may be spelling mistakes in the tweets, these should be corrected to reduce dimensionality/complexity and improve accuracy. 

```python 
from spellchecker import SpellChecker

spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
    
df_train['text']=df_train['text'].apply(lambda x : correct_spellings(x))
df_test['text'] = df_test['text'].apply(lambda x: correct_spellings(x))
```

#### Removing Stop Words
Finally, we remove stop words such as "in", "on", "with", "by" and "for" since these do not provide much information on the sentiment of tweets and will only add unnecessary complexity to our models.

First, we create a string version containing a list of words for further EDA,

```python
df_train['temp_list'] = df_train['text'].apply(lambda x:str(x).split())

# Str version for EDA
def remove_stopword(x):
    return [y for y in x if y not in stopwords.words('english')]
df_train['temp_list'] = df_train['temp_list'].apply(lambda x:remove_stopword(x))
```

we also create a text version which will be used for TF-IDF vectorisation later.

```python 
# Text version for tfidf vectorisation
import nltk
nltk.download('stopwords')
def remove_stopword(x):
    new = ""
    for y in x.split():
        if y not in stopwords.words('english'):
            new = f'{new}{y} '
            #new = new + y + " "
    return new.rstrip()
df_train['text'] = df_train['text'].apply(lambda x:remove_stopword(x))
df_test['text'] = df_test['text'].apply(lambda x:remove_stopword(x))
```

### Post-Preprocessing Data Exploration
Now that preprocessing has been done we have a look at the most common words in each sentiment class. First, lets separate the sentiments.

```python 
Pt_sent = df_train[df_train['sentiment']==1]
Ng_sent = df_train[df_train['sentiment']==-1]
Nt_sent = df_train[df_train['sentiment']==0]
``` 

Now we look at the most common words for each class.

```python
from collections import Counter
# Most common words for each sentiment
for sentiment in [(Pt_sent, 'Greens'), (Nt_sent, 'Blues'), (Ng_sent, 'Reds')]:
    top = Counter([item for sublist in sentiment[0]['temp_list'] for item in sublist])
    temp_sent = pd.DataFrame(top.most_common(20))
    temp_sent.columns = ['Common_words','count']
    display(temp_sent.style.background_gradient(cmap=sentiment[1]))
```

![image](https://user-images.githubusercontent.com/98208084/210032590-65b9288a-3dde-4ba9-b0dc-a90f9d666019.png)

### Modeling
#### TF-IDF
TF-IDF (Term Frequency-Inverse Document Frequency) is a common technique used in natural language processing (NLP) to represent the importance of words in a document. It is typically used to transform text data into numerical vectors that can be used as input to machine learning models. 

We will be applying this vectorizer to our pre-processed data to extract features for our machine learning models. 

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1024)
X_train = vectorizer.fit_transform(df_train['text'])
X_train = np.array(X_train.toarray())
y_train = np.array(df_train['sentiment'])

X_test = vectorizer.transform(df_test['text'])
X_test = np.array(X_test.toarray())
y_test = np.array(df_test['sentiment'])
```

### Logistic Regression 
The first model we will be training is the logistic regression. Logistic regression is a generalised linear model and assumes that the data follows a Bernoulli     distribution. Logistic regression solves for the parameters by maximising the likelihood function and applying gradient descent. The structure of the logistic regression model is simple and interpretable, and the influence of different features on the final results can be seen from the weights of the features.

In logistic regression, the model estimates the probability that an instance belongs to a class (e.g., 0 or 1) using a logistic function, which is defined as:

p = 1 / (1 + e^(-z))

where p is the probability that the instance belongs to the positive class, e is the base of the natural logarithm (approximately 2.718), and z is the linear combination of the features and their weights:

z = w_0 + w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n

where w_0 is the intercept term and w_1, w_2, ..., w_n are the weights of the features x_1, x_2, ..., x_n, respectively.

The logistic function maps the output of the linear combination of the features to a value between 0 and 1, which can be interpreted as the probability that the instance belongs to the positive class. The class is then predicted based on a threshold probability, typically 0.5. If the predicted probability is greater than or equal to the threshold, the instance is classified as the positive class, otherwise it is classified as the negative class.





















