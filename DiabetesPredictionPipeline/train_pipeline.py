import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib
import pipeline
import config

def run_training():
    """Train the model"""

    #read_training_data
    data = pd.read_csv(config.TRAINING_DATA_FILE)

    #divide train and test
    X_train, X_test , y_train, y_test = train_test_split(data.drop(config.TARGET,axis=1),data[config.TARGET],
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=data[config.TARGET]
                                                    )
    
    #pipeline

    pipeline.diabetes_pipeline.fit(X_train,y_train)
    joblib.dump(pipeline.diabetes_pipeline,config.MODEL_PATH)

if __name__ == '__main__':
    run_training()
