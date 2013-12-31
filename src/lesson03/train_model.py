import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_insult_model():
	train = pd.read_csv('train-utf8.csv')

	vectorizer = CountVectorizer()
	X_train = vectorizer.fit_transform(train.Comment)

	model = MultinomialNB().fit(X_train, list(train.Insult))
	return model, vectorizer

def get_prob_of_insult(text, vec, model):
	test = vec.transform([text])
	return model.predict_proba(test)[0][1]


if __name__ == '__main__':
	model, vec = train_insult_model()
	test = vec.transform(["I hate you so fucking much"])
	print model.predict_proba(test)
