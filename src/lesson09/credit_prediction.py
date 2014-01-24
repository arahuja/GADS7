import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

class CreditScoreFeaturizer():
  def __init__(self):
    vectorizer = None

    #Create imputer to fill in missing values
    self._imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

    #A binarizer pushes continous values into two buckets
    self._binarizer = preprocessing.Binarizer()

    #Scalars transform all values to a set scale, this one mapes the [min , max] --> [0, 1]
    self._scaler = preprocessing.MinMaxScaler()
    self._preprocs = [
                      self._imputer, \
                      #self._binarizer, \
                      #self._scaler
                      ]

  def _fit_transform(self, dataset):
    """
      Take the initial dataframe "train" and apply the data processors, like the imputer or binarizer
    """
    for p in self._preprocs:
      dataset = self._proc_fit_transform(p, dataset)
    return dataset

  def _transform(self, dataset):
    """
      Given that you have already trained your data processers, apply the transformations to a new dataset
      This is necessary to create predictions, you create your features based on the training set, but still
      need to transform your heldout, unseen prediction set into a matrix with the same features and scale
    """
    for p in self._preprocs:
      dataset = p.transform(dataset)

    return dataset

  def _proc_fit_transform(self, p, dataset):
    """
      Helper function to "train" and apply a single preprocessing agent

    """
    p.fit(dataset)
    dataset = p.transform(dataset)
    return dataset
    

  def create_features(self, dataset, training=False):
    """
      Transform a datframe <dataset> into a feature matrix

      params:
      dataset : Pandas DataFrame, the input dataset
      training: boolean, flag - True if you want to "learn/train" your preprocessors based on this dataset.

    """

    ###First step, select some fields we care about, all of these are numeric, so we can just pick them out
    ###we don't need a vectorizer ... yet
    data = dataset[['RevolvingUtilizationOfUnsecuredLines', \
                    'age', \
                    'NumberOfTime30-59DaysPastDueNotWorse', \
                    'DebtRatio', \
                    'MonthlyIncome',\
                    'NumberOfOpenCreditLinesAndLoans', \
                    'NumberOfTimes90DaysLate', \
                    'NumberRealEstateLoansOrLines', \
                    'NumberOfTime60-89DaysPastDueNotWorse', \
                    'NumberOfDependents' \
                  ]]

   
    data.NumberOfDependents.fillna(0)
    if training:
      ### If training flag is set, train the preprocessors based on this data
      data = self._fit_transform(data)
    else:
      ### if not training, then apply the preprocessors we already learned
      data = self._transform(data)
    return data

def train_model(X, y):
  """
    Trains the specified model

    X, matrix - input features in matrix format
    y, column array - target variable or outcome

  """

  ## Use any model that we might find appropriate
  #model = RidgeClassifierCV(alphas=[ 0.1, 1., 10. ])

  ##Create the object and set relevant parameters
  model = GradientBoostingClassifier(n_estimators=50)

  #Fit the model
  model.fit(X, y)

  return model

def predict(model, testX):
  """
    Return predictions on new dataset <testX>

    testX : matrix, dataset transformed into feature matrix

  """
  return model.predict(testX)



def create_submission(model, transformer):
  submission_test = pd.read_csv('cs-test.csv')
  predictions = pd.Series(x[1] for x in model.predict_proba(transformer.create_features(submission_test)))


  submission = pd.DataFrame({'Id': submission_test['Unnamed: 0'], 'Probability': predictions})
  submission.sort_index(axis=1, inplace=True)
  submission.to_csv('submission.csv', index=False)



def main():
  data = pd.read_csv('cs-training.csv')
  featurizer = CreditScoreFeaturizer()
  
  print "Transforming dataset into features..."
  ##Create matrix of features from raw dataset
  X = featurizer.create_features(data, training=True)

  ##Set target variable y
  y = data.SeriousDlqin2yrs

  print "Training model..."
  model = train_model(X,y)

  print "Cross validating..."
  print np.mean(cross_val_score(model, X, y, scoring='roc_auc', cv=3))


  print "Create predictions on submission set..."
  create_submission(model, featurizer)


if __name__ == '__main__':
  main()