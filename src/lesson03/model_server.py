from flask import Flask
from flask import request
import train_model


model, vectorizer = train_model.train_insult_model()
app = Flask(__name__)

@app.route("/")
def hello():
	return "Hello World"

@app.route("/insultcheck")
def insult_check():
	sent = request.args.get('q', '')
	print sent
	test = vectorizer.transform([sent])
	probability = model.predict_proba(test)[0][1]
	if probability > 0.5:
		return "That is an insult"
	else:
		return "No, that's clean"

if __name__ == '__main__':
	app.run()
