import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class Preprocessor:

    def __init__(self):
        self.data_expression_median = None
        self.data_CNA = None
        self.data_mutations_mskcc = None
        self.data_clinical_patient = None

        # if there is no data files uncomment the following code.
        # self.get_data_file()

        self.preprocess_clinical_data_file()

    def get_data_file(self):
        data_expression_median = "1jV89yYOUBWnPPC4oOhJGo_MaIwPQ2Nk0"
        data_CNA = "1ac045VSEWOxgkSdZREdW3z5KNRpoObCt"
        data_mutations_mskcc  = "1YOg6G0DGQu5LLIO8-E1Fkz2f43_zNIG-"
        data_clinical_patient = "1GYWvq1XsCqJvTE38hWU-Zu7EMjoKoKnl"
        gdd.download_file_from_google_drive(file_id=data_expression_median, dest_path='../data/data_expression_median.txt')
        gdd.download_file_from_google_drive(file_id=data_CNA, dest_path='../data/data_CNA.txt')
        gdd.download_file_from_google_drive(file_id=data_mutations_mskcc, dest_path='../data/data_mutations_mskcc.txt')
        gdd.download_file_from_google_drive(file_id=data_clinical_patient, dest_path='../data/data_clinical_patient.txt')

    def preprocess_clinical_data_file(self):
        self.data_clinical_patient = pd.read_csv("../data/data_clinical_patient.txt", sep='\t', skiprows=4,index_col=0)
        print(self.data_clinical_patient.head())

        # Deal with missing value
        # delete the missing value row
        # delete the missing value row of categorical value
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['CELLULARITY'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['CHEMOTHERAPY'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['ER_IHC'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['HER2_SNP6'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['HORMONE_THERAPY'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['INFERRED_MENOPAUSAL_STATE'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['INTCLUST'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['OS_STATUS'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['CLAUDIN_SUBTYPE'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['THREEGENE'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['VITAL_STATUS'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['LATERALITY'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['RADIO_THERAPY'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['HISTOLOGICAL_SUBTYPE'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['BREAST_SURGERY'])]

        # delete the missing value row of numeric value
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['LYMPH_NODES_EXAMINED_POSITIVE'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['NPI'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['COHORT'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['AGE_AT_DIAGNOSIS'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['OS_MONTHS'])]

        print(self.data_clinical_patient.head())

        # encode the categorical value
        labelencoder = LabelEncoder()
        self.data_clinical_patient['CELLULARITY'] = labelencoder.fit_transform(self.data_clinical_patient['CELLULARITY'])
        self.data_clinical_patient['CHEMOTHERAPY'] = labelencoder.fit_transform(self.data_clinical_patient['CHEMOTHERAPY'])
        self.data_clinical_patient['ER_IHC'] = labelencoder.fit_transform(self.data_clinical_patient['ER_IHC'])
        self.data_clinical_patient['HER2_SNP6'] = labelencoder.fit_transform(self.data_clinical_patient['HER2_SNP6'])
        self.data_clinical_patient['HORMONE_THERAPY'] = labelencoder.fit_transform(self.data_clinical_patient['HORMONE_THERAPY'])
        self.data_clinical_patient['INFERRED_MENOPAUSAL_STATE'] = labelencoder.fit_transform(self.data_clinical_patient['INFERRED_MENOPAUSAL_STATE'])
        self.data_clinical_patient['INTCLUST'] = labelencoder.fit_transform(self.data_clinical_patient['INTCLUST'])
        self.data_clinical_patient['OS_STATUS'] = labelencoder.fit_transform(self.data_clinical_patient['OS_STATUS'])
        self.data_clinical_patient['CLAUDIN_SUBTYPE'] = labelencoder.fit_transform(self.data_clinical_patient['CLAUDIN_SUBTYPE'])
        self.data_clinical_patient['THREEGENE'] = labelencoder.fit_transform(self.data_clinical_patient['THREEGENE'])
        self.data_clinical_patient['VITAL_STATUS'] = labelencoder.fit_transform(self.data_clinical_patient['VITAL_STATUS'])
        self.data_clinical_patient['LATERALITY'] = labelencoder.fit_transform(self.data_clinical_patient['LATERALITY'])
        self.data_clinical_patient['RADIO_THERAPY'] = labelencoder.fit_transform(self.data_clinical_patient['RADIO_THERAPY'])
        self.data_clinical_patient['HISTOLOGICAL_SUBTYPE'] = labelencoder.fit_transform(self.data_clinical_patient['HISTOLOGICAL_SUBTYPE'])
        self.data_clinical_patient['BREAST_SURGERY'] = labelencoder.fit_transform(self.data_clinical_patient['BREAST_SURGERY'])

        onehotencoder = OneHotEncoder(categorical_features=[2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19])
        self.data_clinical_patient = onehotencoder.fit_transform(self.data_clinical_patient).toarray()

        # Feature Scaling
        sc = StandardScaler()
        self.data_clinical_patient = sc.fit_transform(self.data_clinical_patient)

        print("Generate clinical data matrix successfully")

if __name__ == '__main__':
    preprocessor = Preprocessor()