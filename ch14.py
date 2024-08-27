import pandas as pd
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load a sample dataset from an online source
url = "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv"
df = pd.read_csv(url, delimiter='\t', header=None, names=['text', 'sentiment'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Support Vector Classifier
model = SVC(kernel='linear')

# Train the model
model.fit(X_train_tfidf, y_train)

# Predict the sentiment on the test data
y_pred = model.predict(X_test_tfidf)

# Print the classification report and accuracy
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Perform sentiment analysis using TextBlob on a sample sentence
sample_sentence = "I love the music in this movie."
analysis = TextBlob(sample_sentence)
print(f"Sentence: {sample_sentence}")
print(f"Sentiment: {analysis.sentiment}")
print("Polarity: {:.2f}, Subjectivity: {:.2f}".format(analysis.sentiment.polarity, analysis.sentiment.subjectivity))