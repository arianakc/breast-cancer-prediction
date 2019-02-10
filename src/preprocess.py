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
        # after row 1984 we cannot use because of two much NAN value
        self.data_clinical_patient = pd.read_csv("../data/data_clinical_patient.txt", sep='\t', skiprows=524)
        print(self.data_clinical_patient.head())
        imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
        



if __name__ == '__main__':
    preprocessor = Preprocessor()