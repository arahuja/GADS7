from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.datasets import fetch_20newsgroups

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

# Load the text data
print "Loading data..."
twenty_train_subset = fetch_20newsgroups(subset='train', categories=categories)


print "Transforming features..."
# Turn the text documents into vectors of word frequencies
# vectorizer = CountVectorizer()
# X_train = vectorizer.fit_transform(twenty_train_subset.data)

# #TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(twenty_train_subset.data)

#TFIDF advanced
from sklearn.feature_extraction.text import TfidfVectorizer

# vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
# X_train = vectorizer.fit_transform(twenty_train_subset.data)

from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2), min_df=1)
X_train = vectorizer.fit_transform(twenty_train_subset.data)


print "Training DecisionTreeClassifier ... this could take a while"
tree_model = DecisionTreeClassifier()
scores = cross_val_score(tree_model, X_train.toarray(), twenty_train_subset.target)
print "Cross validation scores..."
print scores

# from sklearn.ensemble import RandomForestClassifier

print "Training RandomForestClassifier ... this could take a while"
rf_model = RandomForestClassifier(n_estimators=10)

scores = cross_val_score(rf_model, X_train.toarray(), twenty_train_subset.target)
print "Cross validation scores..."
print scores

rf_model.fit(X_train.toarray(), twenty_train_subset.target)


#This prints the top 10 most important features
print sorted(zip(rf_model.feature_importances_, vectorizer.get_feature_names()), reverse=True)[:10]

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=5)
print "Training AdaBoostClassifier ... this could take a while"
scores = cross_val_score(model, X_train.toarray(), twenty_train_subset.target)
print "Cross validation scores..."
print scores

## DOESN'T WORK
# from sklearn.linear_model import LogisticRegression
# model = AdaBoostClassifier(n_estimators=5, base_estimator=LogisticRegression())
# print cross_val_score(model, X_train.toarray(), twenty_train_subset.target)
