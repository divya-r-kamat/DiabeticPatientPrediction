import pandas as pd

import joblib
import config
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
#from sklearn.metrics.scorer import make_scorer

def make_prediction(input_data):
    _diabetes_pipe = joblib.load(filename=config.MODEL_PATH)
    results = _diabetes_pipe.predict(input_data)
    return results

if __name__ == '__main__':
    #test pipeline

    #read_training_data
    data = pd.read_csv(config.TRAINING_DATA_FILE)

    #divide train and test
    X_train, X_test , y_train, y_test = train_test_split(data.drop(config.TARGET,axis=1),data[config.TARGET],
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=data[config.TARGET]
                                                    )

    y_pred = make_prediction(X_test)


    score = 100 * recall_score(y_test,y_pred,average='weighted')
    print('Test recall:',score)
