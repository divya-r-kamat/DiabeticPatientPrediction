#data
TRAINING_DATA_FILE = 'dataset/train.csv'
PIPELINE_NAME = 'pipeline_logistic_regression'
MODEL_PATH = 'model/pipeline_logistic_regression'

#Configuration variables
TARGET = 'diabetesMed'

DROP_FEATURES = ['weight','patient_id','tel_20','tel_28','tel_29','tel_30','tel_41','tel_45','tel_47','tel_2','encounter_id']#,'tel_46','tel_23','tel_43'

CATEGORICAL_VARS_WITH_NA = ['race','tel_1','tel_9','tel_10','tel_11']

CATEGORICAL_VARS_TO_LABELENCODE =['tel_9','tel_10','tel_11'] 
#RARE_ENCODE_5PCT_TRESH = ['tel_1','discharge_disposition_id','admission_source_id']

NUM_VARS = ['time_in_hospital','tel_3','tel_4','tel_5','tel_6','tel_7','tel_8','tel_12']
CATEGORICAL_VARS = ['race', 'gender', 'tel_1','tel_13', 'tel_14','tel_48' , 'tel_49']  
DROP_TRANSFORMED_FEATURES=['race','gender','tel_13','tel_14','tel_48','tel_49']

TEST_COLUMNS = ['tel_15','tel_16','tel_17','tel_18','tel_19','tel_21','tel_22','tel_23','tel_24','tel_25','tel_26','tel_27','tel_42','tel_43','tel_44','tel_46']