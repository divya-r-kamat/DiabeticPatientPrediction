import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import LabelEncoder


#Rare label imputer
class RareLabelCategoricalEncoder(BaseEstimator,TransformerMixin):
    def __init__(self,tot=0.05,variables=None):
        self.tot = tot
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        self.encoder_dict_ = {}
        for var in self.variables:
            t = pd.Series(X[var].value_counts()/np.float(len(X)))  
            self.encoder_dict_[var] = list(t[t>=self.tot].index)
            return self

    def transform(self,X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]),X[feature],'Rare')  
        return X

#One hot encoding using pd.get_dummies
class ToDummiesTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X = X.copy()
        X_temp = pd.get_dummies(X[self.variables],drop_first=True)
        #X = X.drop(X[self.variables],axis=1)
        #X = pd.concat([X_temp,X],axis=1)
        return X_temp


#Drop uncessary features
class DropFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
      return self

    def transform(self,X):
      X = X.copy()
      for feature in self.variables:
        X.drop(feature,axis=1,inplace=True)
      return X

#categorical missing value imputer
class CategoricalImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')
        return X

#custom imputer to encode tel_9,tel_10 and tel_11 variables
class TelCustomImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X = X.copy()
        for feature in self.variables:
               X.loc[X[feature].astype(str).str.contains('E'),feature] = 'E'
               X.loc[X[feature].astype(str).str.contains('V'),feature] = 'V'
               X.loc[X[feature].astype(str).str.contains('[0-9]'),feature] = 'NUM'
        return X[self.variables]


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, df):
        return df[self.key]

#categorical custom imputer to replace test columns       
class EncodeTelCustomImputer(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].replace('No',0)
            X[feature] = X[feature].replace('Steady',1)
            X[feature] = X[feature].replace('Up',1)
            X[feature] = X[feature].replace('Down',1)
        return  X[self.variables]

#categorical custom feature engineering for test columns       
class CustomTelFeature(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]           
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X = X.copy()
        X['tot_test_columns'] = 0
        for feature in self.variables:
            X['tot_test_columns'] = X['tot_test_columns'] + X[feature]
        return  X[['tot_test_columns']]

#categorical custom feature engineering for age columns       
class CustomAgeFeature(BaseEstimator,TransformerMixin):
    def __init__(self,variable=None):
            self.variable = variable

    def fit(self,X,y=None):
        return self

    def transform(self,X):
      X = X.copy()
      age_id = {'[0-10)':5, 
          '[10-20)':15, 
          '[20-30)':25, 
          '[30-40)':35, 
          '[40-50)':45, 
          '[50-60)':55,
          '[60-70)':65, 
          '[70-80)':75, 
          '[80-90)':85, 
          '[90-100)':95}
      X['age_group'] = X[self.variable].replace(age_id)
      return  X[['age_group']]

class MultiColumnLabelEncoder(BaseEstimator,TransformerMixin):    
    def __init__(self, columns = None):
        self.columns = columns # list of column to encode
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        
        output = X.copy()
        
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        
        return output[self.columns]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)