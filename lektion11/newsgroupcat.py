categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

# Load the list of files matching those categories as follows:
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(
                  subset='train', categories=categories,
                  shuffle=True, random_state=42)

# Did we get what we wanted?
twenty_train.target_names

# Size
len(twenty_train.data)

# Lets see what we have :
print("\n".join(twenty_train.data[0].split("\n")[:2]))

print(twenty_train.target_names[twenty_train.target[0]])

print("\n".join(twenty_train.data[1000].split("\n")[:2]))

print(twenty_train.target_names[twenty_train.target[1000]])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english')

X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()


X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Import the SKlearn model we are using
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                      hidden_layer_sizes=(10, 10,10), random_state=1)
clf.fit(X_train_tfidf, twenty_train.target)


docs_new = ['Doctors are bad', 'OpenGL on the GPU is fast']

X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)


predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
   print('%r => %s' % (doc, twenty_train.target_names[category]))

twenty_test = fetch_20newsgroups(
     subset='test', categories=categories,
     shuffle=True, random_state=42)

from sklearn.pipeline import Pipeline
text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10,10), random_state=1)),
         ])

text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(twenty_test.data)


import numpy as np
np.mean(predicted == twenty_test.target)