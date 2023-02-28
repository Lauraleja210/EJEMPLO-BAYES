# PARTE 1
# Terminal - pip install requests
# pip install pandas 
# pip install nlkt
# pip install sklearn.model  
import requests 
import zipfile
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
data_file = 'SMSSpamCollection'

# Make request
resp = requests.get(url)

# Get filename
filename = url.split('/')[-1]

# Download zipfile
with open(filename, 'wb') as f:
  f.write(resp.content)

# Extract Zip
with zipfile.ZipFile(filename, 'r') as zip:
  zip.extractall('')

# Read Dataset
data = pd.read_table(data_file, 
                     header = 0,
                     names = ['type', 'message']
                     )

# Show dataset
data.head()


#PARTE 2
import nltk

# Install everything necessary
nltk.download('punkt')
nltk.download('stopwords')


from nltk.stem.porter import *
from nltk.corpus import stopwords
stop = stopwords.words('english')

# Tokenize
data['tokens'] = data.apply(lambda x: nltk.word_tokenize(x['message']), axis = 1)

# Remove stop words
data['tokens'] = data['tokens'].apply(lambda x: [item for item in x if item not in stop])

# Apply Porter stemming
stemmer = PorterStemmer()
data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(item) for item in x])

#PARTE 3

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Unify the strings once again
data['tokens'] = data['tokens'].apply(lambda x: ' '.join(x))

# Make split
x_train, x_test, y_train, y_test = train_test_split(
    data['tokens'], 
    data['type'], 
    test_size= 0.2
    )

# Create vectorizer
vectorizer = CountVectorizer(
    strip_accents = 'ascii', 
    lowercase = True
    )

# Fit vectorizer & transform it
vectorizer_fit = vectorizer.fit(x_train)
x_train_transformed = vectorizer_fit.transform(x_train)
x_test_transformed = vectorizer_fit.transform(x_test)

#PARTE 4

# Build the model
from sklearn.naive_bayes import MultinomialNB

# Train the model
naive_bayes = MultinomialNB()
naive_bayes_fit = naive_bayes.fit(x_train_transformed, y_train)

from sklearn.metrics import confusion_matrix, balanced_accuracy_score

# Make predictions
train_predict = naive_bayes_fit.predict(x_train_transformed)
test_predict = naive_bayes_fit.predict(x_test_transformed)

def get_scores(y_real, predict):
  ba_train = balanced_accuracy_score(y_real, predict)
  cm_train = confusion_matrix(y_real, predict)

  return ba_train, cm_train 

def print_scores(scores):
  return f"Balanced Accuracy: {scores[0]}\nConfussion Matrix:\n {scores[1]}"

train_scores = get_scores(y_train, train_predict)
test_scores = get_scores(y_test, test_predict)


print("## Train Scores")
print(print_scores(train_scores))
print("\n\n## Test Scores")
print(print_scores(test_scores))

