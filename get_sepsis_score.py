#!/usr/bin/env python
from sklearn.externals import joblib
import numpy as np
import pandas as pd

def get_sepsis_score(data, model):
    num_rows = len(data)

    meanF = np.load('meanF.npy')
    M1 = joblib.load('model-saved.pkl')

    ####### Impute
    imputePatient = []
    for interval in range(data.shape[0]):      #### loop for on intervals
        if interval == 0:
            newData = np.copy(data[0,:])
            for column in range(40):       ########  loop for on columns
                if (np.isnan(newData[column])):
                    newData[column] = meanF[column]
            # imputePatient.append(newData)
            imputePatient = newData

        else:
            index = np.arange(interval+1)
            aa = np.copy(data[index])
            aa[0, :] = newData
            df = pd.DataFrame.from_records(aa)
            df.interpolate(method='linear', inplace=True)
            newData1 = np.array(df)
            # imputePatient.append(newData1[-1, :])
            imputePatient = np.vstack((imputePatient, newData1[-1, :]))

    data = imputePatient
    ####### End Impute
    if num_rows==1:
        label = 0.0
        score = 0.4
    else:
        predicted = M1.predict(data)
        if predicted[num_rows - 1] == 0:
            score = 0.4
        else:
            score = 0.6
        label = predicted[num_rows - 1]

    # ###################################

    # return score_final, label_final


    return score, label

def load_sepsis_model():

    return None
