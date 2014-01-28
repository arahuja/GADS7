import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

class LemonCarFeaturizer():
  def __init__(self):
    vectorizer = None
    self._imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    self._binarizer = preprocessing.Binarizer()
    self._scaler = preprocessing.StandardScaler()
    self._preprocs = [self._imputer,
                      #self._scaler, 
                      #self._binarizer 
                      ]

  def _fit_transform(self, dataset):
    for p in self._preprocs:
      dataset = self._proc_fit_transform(p, dataset)
    return dataset

  def _transform(self, dataset):
    for p in self._preprocs:
      dataset = p.transform(dataset)

    return dataset

  def _proc_fit_transform(self, p, dataset):
    p.fit(dataset)
    dataset = p.transform(dataset)
    return dataset

  def create_features(self, dataset, training=False):
    data = dataset[ [ 
                  'VehOdo',
                  'VehYear',
                  'MMRAcquisitonRetailCleanPrice', 
                  'MMRCurrentAuctionAveragePrice', 
                  'MMRCurrentAuctionCleanPrice',
                  'MMRCurrentRetailAveragePrice', 
                  'MMRCurrentRetailCleanPrice'] 
          ]

    data['MPY'] = data.VehOdo/ (data.VehYear - data.VehYear.min())
    actype = pd.get_dummies(dataset['AUCGUART'])
    data = pd.concat([data, actype], axis=1)

    print data.head()

    if training:
      data = self._fit_transform(data)
    else:
      data = self._transform(data)
    return data

def train_model(X, y):
  #model = GradientBoostingClassifier(n_estimators=50)
  #model = RidgeClassifierCV(alphas=[ 0.1, 1., 10. ])
  model = LogisticRegression()
  #model = DecisionTreeClassifier() 
  model.fit(X, y)
  #print model.coef_
  return model

def predict(model, y):
  return model.predict(y)

def create_submission(model, transformer):
  submission_test = pd.read_csv('inclass_test.csv')
  predictions = pd.Series([x[1]
    for x in model.predict_proba(transformer.create_features(submission_test))])

  submission = pd.DataFrame({'RefId': submission_test.RefId, 'IsBadBuy': predictions})
  #submission.sort_index(axis=1, inplace=True)
  submission.to_csv('submission.csv', index=False)



def main():
  data = pd.read_csv('inclass_training.csv')
  featurizer = LemonCarFeaturizer()
  
  print "Transforming dataset into features..."
  X = featurizer.create_features(data, training=True)
  y = data.IsBadBuy

  print "Training model..."
  model = train_model(X,y)

  print "Cross validating..."
  print np.mean(cross_val_score(model, X, y, scoring='roc_auc'))

  print "Create predictions on submission set..."
  create_submission(model, featurizer)


if __name__ == '__main__':
  main()
