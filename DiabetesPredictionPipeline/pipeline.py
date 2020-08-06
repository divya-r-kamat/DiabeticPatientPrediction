import pandas as pd
import numpy as np

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import preprocessors as pp 
import config



prep_data = Pipeline(steps = [('drop_uncessary_features', pp.DropFeatures(variables=config.DROP_FEATURES)),
     ('categorical_imputer', pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA))])


custom_impute = Pipeline(steps = [('prep_data', prep_data),
                           ('custom_imputer',pp.TelCustomImputer(variables=config.CATEGORICAL_VARS_TO_LABELENCODE)),
                           ('label_encode',pp.MultiColumnLabelEncoder(columns=config.CATEGORICAL_VARS_TO_LABELENCODE)),
                           ('onehot_encode',pp.ToDummiesTransformer(variables=config.CATEGORICAL_VARS_TO_LABELENCODE))
                           ])

encode_categorical_col = Pipeline(steps = [('prep_data', prep_data),
                           ('onehot_encode',pp.ToDummiesTransformer(variables=config.CATEGORICAL_VARS))
                           ])

age_feature = Pipeline(steps = [('prep_data', prep_data),
                           ('cus_age_feat',pp.CustomAgeFeature(variable='age'))
                           ])

custom_encode_tel = Pipeline(steps = [('prep_data', prep_data),
                           ('encode_tel',pp.EncodeTelCustomImputer(variables=config.TEST_COLUMNS))
                           ])

cust_tel_feature = Pipeline(steps = [('custom_encode_tel',custom_encode_tel),
                           ('cus_tel_feat',pp.CustomTelFeature(variables=config.TEST_COLUMNS))
                           ])


union = FeatureUnion([
                      ('custom_impute',custom_impute),
                      ('encode_categorical',encode_categorical_col),
                      ('custom_encode_tel',custom_encode_tel),
                      ('cust_tel_feature',cust_tel_feature),
                      ('age_feature',age_feature)
                    ])

diabetes_pipeline = Pipeline(
        [('union', union),
          ('scaler',StandardScaler()),
          ('logreg', LogisticRegression())
         ])
