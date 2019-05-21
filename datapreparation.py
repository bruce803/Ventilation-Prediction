import pandas as pd
import matplotlib as plt
import numpy as np
import itertools
import operator
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
try:
    import cPickle as pickle
except:
    import pickle


# def data_preparation(dataFrame):
#     if isinstance(dataFrame, pd.DataFrame):
#
#         data_list = []
#         icustay_id = dataFrame.icustay_id.unique()
#
#         # normailiza the data
#         df_fea = dataFrame.iloc[:, 5:27]
#         feaMatrix = df_fea.values
#         min_max_scaler = preprocessing.MinMaxScaler()
#         feaMatrix_scaled = min_max_scaler.fit_transform(feaMatrix)
#         feaMatrix_scaled_df = pd.DataFrame(feaMatrix_scaled)
#
#         # add columns: 'icustay_id', 'ventilation'
#         icustay = dataFrame.iloc[:, 1]
#         ventilation = dataFrame.iloc[:, 27]
#
#         feaMatrix_scaled_df['icustay_id'] = icustay.values
#         feaMatrix_scaled_df['ventilation'] = ventilation.values
#
#         for pid in icustay_id:
#
#             # feature matrix
#             patient_feature_df = feaMatrix_scaled_df[feaMatrix_scaled_df['icustay_id'] == pid].iloc[:,
#                                  0:22]  # feature area 0:22
#             patient_feature_array = patient_feature_df.values
#
#             # event indicator and event time
#             this_patient = feaMatrix_scaled_df[
#                 feaMatrix_scaled_df.icustay_id == pid]  # iterate patients with their icustay_id
#             this_patient_venti = this_patient.ventilation.values  # return the ventilation flag as array
#
#             '''check the flag array'''
#             flag_check = np.nonzero(this_patient_venti)  # return a tuple with only 1 array element
#
#             if patient_feature_array.shape[0] >= 715:
#                 if flag_check[0].size == 0:  # censored
#                     time = this_patient_venti.size
#                     data_list.append((patient_feature_array, 0, time))
#                 else:
#                     time = flag_check[0][0]
#                     data_list.append((patient_feature_array, int(1), time))
#             else:
#                 if flag_check[0].size == 0:  # censored
#                     time = this_patient_venti.size
#                     patient_feature_fill0 = np.concatenate(
#                         (np.zeros((715 - patient_feature_array.shape[0], 22)), patient_feature_array))
#                     data_list.append((patient_feature_fill0, 0, time))
#                 else:
#                     time = flag_check[0][0]
#                     patient_feature_fill0 = np.concatenate(
#                         (np.zeros((715 - patient_feature_array.shape[0], 22)), patient_feature_array))
#                     data_list.append((patient_feature_fill0, int(1), time))
#
#     return data_list


def data_preparation(dataFrame, seq_len):

    assert isinstance(dataFrame, pd.DataFrame), \
        "the input is not a Padas DataFrame"

    data_list = []
    icustay_id = dataFrame.icustay_id.unique()

    # normailiza the data
    df_fea = dataFrame.iloc[:, 5:27]
    feaMatrix = df_fea.values
    min_max_scaler = preprocessing.MinMaxScaler()
    feaMatrix_scaled = min_max_scaler.fit_transform(feaMatrix)
    feaMatrix_scaled_df = pd.DataFrame(feaMatrix_scaled)

    # add columns: 'icustay_id', 'ventilation'
    icustay = dataFrame.iloc[:, 1]
    ventilation = dataFrame.iloc[:, 27]

    feaMatrix_scaled_df['icustay_id'] = icustay.values
    feaMatrix_scaled_df['ventilation'] = ventilation.values

    for pid in icustay_id:

        # feature matrix
        patient_feature_df = feaMatrix_scaled_df[feaMatrix_scaled_df['icustay_id'] == pid].iloc[:,
                             0:22]  # feature area 0:22
        patient_feature_array = patient_feature_df.values
        # print(patient_feature_array.shape)

        # event indicator and event time
        this_patient = feaMatrix_scaled_df[
            feaMatrix_scaled_df.icustay_id == pid]  # iterate patients with their icustay_id
        this_patient_venti = this_patient.ventilation.values  # return the ventilation flag as array

        '''check the flag array'''
        flag_check = np.nonzero(this_patient_venti)  # return a tuple with only 1 array element

        if patient_feature_array.shape[0] >= seq_len:
            if flag_check[0].size == 0:  # censored
                time = this_patient_venti.size
                data_list.append((patient_feature_array[0:seq_len,:], 0, time))
            else:
                time = flag_check[0][0]
                data_list.append((patient_feature_array[0:seq_len], int(1), time))
        else:
            if flag_check[0].size == 0:  # censored
                time = this_patient_venti.size
                patient_feature_fill0 = np.concatenate(
                    (np.zeros((seq_len - patient_feature_array.shape[0], 22)), patient_feature_array))
                data_list.append((patient_feature_fill0, 0, time))
            else:
                time = flag_check[0][0]
                patient_feature_fill0 = np.concatenate(
                    (np.zeros((seq_len - patient_feature_array.shape[0], 22)), patient_feature_array))
                data_list.append((patient_feature_fill0, int(1), time))

    return data_list


# mimic = 'mimic.csv'
# mimic3 = pd.read_csv(mimic)
#
# print('###### Begin to dump file...')
#
# sample_list = data_preparation(mimic3,30)
#
#
# with open('train_30.pkl', 'wb') as fp:
#     pickle.dump(sample_list[0:20000], fp)
#
# with open('test_30.pkl', 'wb') as file:
#     pickle.dump(sample_list[20000:25052], file)

