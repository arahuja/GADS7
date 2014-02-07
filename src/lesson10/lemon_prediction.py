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
                  'VehicleAge',
                  'VehBCost',
                  'IsOnlineSale',
                  'MMRAcquisitionAuctionAveragePrice',
                  'MMRAcquisitionAuctionCleanPrice',
                  'MMRAcquisitionRetailAveragePrice', 
                  'MMRAcquisitonRetailCleanPrice', 
                  'MMRCurrentAuctionAveragePrice', 
                  'MMRCurrentAuctionCleanPrice',
                  'MMRCurrentRetailAveragePrice', 
                  'VNZIP1',
                  'WarrantyCost', 
                  'MMRCurrentRetailCleanPrice'] 
          ]

    data['MPY'] = data.VehOdo/ (data.VehYear - data.VehYear.min())

    # diff between: 
    # Acquisition price for this vehicle in average condition at time of purchase 
    # Acquisition price for this vehicle in average condition as of current day
    data['priceDiffRetailAuction'] = data.MMRAcquisitionRetailAveragePrice - data.MMRAcquisitionAuctionAveragePrice
    
    # Acquisition price for this vehicle in the above condition as of current day
    # 
    data['priceDiff2'] = data.MMRCurrentAuctionCleanPrice - data.MMRAcquisitionAuctionCleanPrice
    
    ##AUCGUART        The level guarntee provided by auction for the 
    # vehicle (Green light - Guaranteed/arbitratable, Yellow Light - caution/issue, 
    # red light - sold as is)
    actype = pd.get_dummies(dataset['AUCGUART'])
    data = pd.concat([data, actype], axis=1)

    #use the size category
    sizeType = pd.get_dummies(dataset['Size'])
    data = pd.concat([data, sizeType], axis=1)

    auctionType = pd.get_dummies(dataset['Auction'])
    data = pd.concat([data, auctionType], axis=1)

    TopThreeType = pd.get_dummies(dataset['TopThreeAmericanName'])
    data = pd.concat([data, TopThreeType], axis=1)

    NationalityType = pd.get_dummies(dataset['Nationality'])
    data = pd.concat([data, NationalityType], axis=1)

    WheelType = pd.get_dummies(dataset['WheelType'])
    data = pd.concat([data, WheelType], axis=1)

    Color = pd.get_dummies(dataset['Color'])
    data = pd.concat([data, Color], axis=1)

    top_make = set( dataset.Make.value_counts().index[:10])
    dataset['Smartmake'] = dataset.Make.map(lambda make: make if make in top_make else 'Other')
    makeType = pd.get_dummies(dataset['Smartmake'])
    data = pd.concat([data, makeType], axis=1)

    # use top ten Models
    top_models = set (dataset.Model.value_counts().index[:50])
    dataset['SmartModel'] = dataset.Model.map(lambda model: model if model in top_models else 'Other')
    modelType = pd.get_dummies(dataset['SmartModel'])
    data = pd.concat([data, modelType], axis=1)


    ## 
    ## this one made it crash because it has 37 possible values
    ## get error of: too many boolean indexes
    ## 
    # StateType = pd.get_dummies(dataset['VNST'])
    # data = pd.concat([data, StateType], axis=1)

    ## this ones decreased the score
    ##

    # putype = pd.get_dummies(dataset['PRIMEUNIT'])
    # data = pd.concat([data,putype], axis=1)
    #top_states = set (dataset.VNST.value_counts().index[:20])
    #dataset['ST'] = dataset.VNST.map(lambda st: st if st in top_states else 'Other')
    #state = pd.get_dummies(dataset['ST'])
    #data = pd.concat([data, state], axis=1)

    # top_10_buyers = set (dataset.BYRNO.value_counts().index[:10])
    # dataset['buyers'] = dataset.BYRNO.map(lambda buyer: buyer if buyer in top_10_buyers else 'Other')
    # buyerType = pd.get_dummies(dataset['buyers'])
    # data = pd.concat([data, buyerType], axis=1)

    # top_10_submodels = set(dataset.SubModel.value_counts().index[:10])
    # dataset['smartsubmodel'] = dataset.SubModel.map(lambda sm: sm if sm in top_10_submodels else 'Other_sm')
    # ssm = pd.get_dummies(dataset['smartsubmodel'])
    # data = pd.concat([data, ssm], axis = 1)

    # top_states = set (dataset.VNST.value_counts().index[:20])
    # dataset['state'] = dataset.VNST.map(lambda st: st if st in top_states else 'Other')
    # state = pd.get_dummies(dataset['state'])
    # data = pd.concat([data, state], axis=1)



    #use
    if training:

      print data.head()
      data = self._fit_transform(data)
    else:
      data = self._transform(data)
    return data

def train_model(X, y):
  model = GradientBoostingClassifier(n_estimators=100, max_depth=5)#200)
  #model = RidgeClassifierCV(alphas=[ 0.1, 1., 10. ])
  #model = LogisticRegression()
  #model = DecisionTreeClassifier() 
  #model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')

  model.fit(X, y)
  #print model.coef_
  return model

def predict(model, y):
  return model.predict(y)

def create_submission(model, transformer):
  submission_test = pd.read_csv('inclass_test.csv')
  #submission_test = pd.read_csv('test.csv')

  predictions = pd.Series([x[1]
    for x in model.predict_proba(transformer.create_features(submission_test))])

  submission = pd.DataFrame({'RefId': submission_test.RefId, 'IsBadBuy': predictions})
  #submission.sort_index(axis=1, inplace=True)
  submission.to_csv('submission.csv', index=False)



def main():
  data = pd.read_csv('inclass_training.csv')
  #data = pd.read_csv('training.csv')
  featurizer = LemonCarFeaturizer()
  
  print "Transforming dataset into features..."
  X = featurizer.create_features(data, training=True)
  y = data.IsBadBuy

  print "Training model..."
  model = train_model(X,y)

  print "Cross validating..."
  score = np.mean(cross_val_score(model, X, y, n_jobs=-1, scoring='roc_auc'))
  print score
  #plot_roc("Your Classifier ", score)

  print "Create predictions on submission set..."
  create_submission(model, featurizer)


if __name__ == '__main__':
  main()
