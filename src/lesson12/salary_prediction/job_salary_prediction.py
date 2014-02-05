import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

from patsy import DesignMatrixBuilder, dmatrices, dmatrix
import scipy 

from sklearn.feature_extraction.text import CountVectorizer

def get_training_set(data):


  data['IsLondon'] = data.LocationNormalized.map( lambda x: "London" in x )
  y, X = dmatrices("SalaryNormalized ~ Category + IsLondon", data)


  ## Adding text_features
  title_transformer = CountVectorizer(ngram_range=(1,2), stop_words='english', max_features=500)
  title_features = title_transformer.fit_transform(data.Title.fillna(""))

  X = scipy.sparse.hstack((X, title_features))

  description_transformer = CountVectorizer(max_features=450, ngram_range=(1,2), stop_words='english')
  description_features = description_transformer.fit_transform(data.FullDescription.fillna(""))
  
  #X = scipy.sparse.hstack((X, description_features))


  location_transformer = CountVectorizer(max_features=250)
  location_features = location_transformer.fit_transform(data.LocationRaw.fillna(""))
  
  X = scipy.sparse.hstack((X, location_features))
  print X.shape
  return X, y



if __name__ == '__main__':
  
  data = pd.read_csv('train.csv')
  X, Y = get_training_set(data)
  Y = np.log(Y)
  Y = np.reshape(Y, (Y.shape[0]))

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

  #model = linear_model.LassoCV(alphas=[0.01, 0.1, 1.0])

  model = linear_model.LinearRegression()
  model.fit(X_train, np.array(Y_train))
  predictions = model.predict(X_test)

  print mean_absolute_error(np.exp(Y_test), np.exp(predictions))

  mae_scorer = make_scorer(mean_absolute_error)
  msqe_scorer = make_scorer(mean_squared_error)

  #print cross_val_score(model, X, Y, scoring='mean_squared_error').mean()
  #print cross_val_score(model, X, Y, scoring=mae_scorer).mean()
  print cross_val_score(model, X, Y).mean()