import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.Baseline import devide


class Preprocessor:

    def __init__(self):
        self.data_expression_median = None
        self.data_CNA = None
        self.data_mutations_mskcc = None
        self.data_clinical_patient = None
        self.patient_dict = {}
        self.genomic_patient_features = []
        self.genomic_X = None
        self.genomic_Y = None
        self.clinical_X = None
        self.clinical_Y = None
        self.clinical_patients = None
        self.clinical_X_genomic_existed = []
        self.clinical_Y_genomic_existed = []
        self.patient_vector_dict_X = {}
        self.patient_vector_dict_Y = {}
        # if there is no data file uncomment the following code.
        # self.get_data_file()

        self.preprocess_clinical_data_file()
        self.preprocess_genomic_data_file()

    def Y_encoding(self, data):
        Y = []
        for i in range(len(data)):
            if data['OS_MONTHS'][i] > 120:
                Y.append(0)  # not affected
            elif data['VITAL_STATUS'][i] == "Died of Disease":
                Y.append(1)  # affected
            else:
                Y.append(-1)  # ignore
        return Y

    def get_rid_of_nulls(self, value):
        if pd.isnull(value):
            return 'Unknown'
        else:
            return value

    def pca(self, X=np.array([]), no_dims=50):
        """
            Runs PCA on the NxD array X in order to reduce its dimensionality to
            no_dims dimensions.
        """
        print("Preprocessing the data using PCA...")
        (n, d) = X.shape
        X = X - np.tile(np.mean(X, 0), (n, 1))
        (l, M) = np.linalg.eig(np.dot(X.T, X))
        Y = np.dot(X, M[:, 0:no_dims])
        return Y

    def lst_one_hot(self, lst, size):
        onehot = np.zeros(size)
        for i in lst:
            onehot[i] = 1
        return list(onehot)

    def get_data_file(self):
        data_expression_median = "1jV89yYOUBWnPPC4oOhJGo_MaIwPQ2Nk0"
        data_CNA = "1ac045VSEWOxgkSdZREdW3z5KNRpoObCt"
        data_mutations_mskcc = "1YOg6G0DGQu5LLIO8-E1Fkz2f43_zNIG-"
        data_clinical_patient = "1GYWvq1XsCqJvTE38hWU-Zu7EMjoKoKnl"
        gdd.download_file_from_google_drive(file_id=data_expression_median,
                                            dest_path='../data/data_expression_median.txt')
        gdd.download_file_from_google_drive(file_id=data_CNA, dest_path='../data/data_CNA.txt')
        gdd.download_file_from_google_drive(file_id=data_mutations_mskcc, dest_path='../data/data_mutations_mskcc.txt')
        gdd.download_file_from_google_drive(file_id=data_clinical_patient,
                                            dest_path='../data/data_clinical_patient.txt')

    def preprocess_clinical_data_file(self):
        self.data_clinical_patient = pd.read_csv("../data/data_clinical_patient.txt", sep='\t', skiprows=4, index_col=0)
        # print(self.data_clinical_patient.head())

        # add Y
        self.data_clinical_patient['Y'] = self.Y_encoding(self.data_clinical_patient)

        # delete the related but not Y column
        self.data_clinical_patient.drop(columns=['OS_MONTHS', 'VITAL_STATUS'], inplace=True)

        # drop ignored rows
        self.data_clinical_patient = self.data_clinical_patient[self.data_clinical_patient['Y'] != -1]

        # drop the missing value rows of numeric value
        self.data_clinical_patient = self.data_clinical_patient[
            pd.notnull(self.data_clinical_patient['LYMPH_NODES_EXAMINED_POSITIVE'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['NPI'])]
        self.data_clinical_patient = self.data_clinical_patient[pd.notnull(self.data_clinical_patient['COHORT'])]
        self.data_clinical_patient = self.data_clinical_patient[
            pd.notnull(self.data_clinical_patient['AGE_AT_DIAGNOSIS'])]

        # Deal with categorical missing value
        # drop the missing value row whose number of missing value is >2:
        self.data_clinical_patient['null_count'] = self.data_clinical_patient.isnull().sum(axis=1)
        self.data_clinical_patient = self.data_clinical_patient[self.data_clinical_patient['null_count'] <= 2]
        self.data_clinical_patient.drop(columns=['null_count'], inplace=True)

        self.clinical_patients = list(self.data_clinical_patient.index.values)

        # set the null value as a categorial
        self.data_clinical_patient['CELLULARITY'] = self.data_clinical_patient['CELLULARITY'].apply(
            self.get_rid_of_nulls)
        self.data_clinical_patient['ER_IHC'] = self.data_clinical_patient['ER_IHC'].apply(self.get_rid_of_nulls)
        self.data_clinical_patient['THREEGENE'] = self.data_clinical_patient['THREEGENE'].apply(self.get_rid_of_nulls)
        self.data_clinical_patient['LATERALITY'] = self.data_clinical_patient['LATERALITY'].apply(self.get_rid_of_nulls)
        self.data_clinical_patient['HISTOLOGICAL_SUBTYPE'] = self.data_clinical_patient['HISTOLOGICAL_SUBTYPE'].apply(
            self.get_rid_of_nulls)
        self.data_clinical_patient['BREAST_SURGERY'] = self.data_clinical_patient['BREAST_SURGERY'].apply(
            self.get_rid_of_nulls)

        # encode the categorical value
        labelencoder = LabelEncoder()
        self.data_clinical_patient['CELLULARITY'] = labelencoder.fit_transform(
            self.data_clinical_patient['CELLULARITY'])
        self.data_clinical_patient['CHEMOTHERAPY'] = labelencoder.fit_transform(
            self.data_clinical_patient['CHEMOTHERAPY'])
        self.data_clinical_patient['ER_IHC'] = labelencoder.fit_transform(self.data_clinical_patient['ER_IHC'])
        self.data_clinical_patient['HER2_SNP6'] = labelencoder.fit_transform(self.data_clinical_patient['HER2_SNP6'])
        self.data_clinical_patient['HORMONE_THERAPY'] = labelencoder.fit_transform(
            self.data_clinical_patient['HORMONE_THERAPY'])
        self.data_clinical_patient['INFERRED_MENOPAUSAL_STATE'] = labelencoder.fit_transform(
            self.data_clinical_patient['INFERRED_MENOPAUSAL_STATE'])
        self.data_clinical_patient['INTCLUST'] = labelencoder.fit_transform(self.data_clinical_patient['INTCLUST'])
        self.data_clinical_patient['OS_STATUS'] = labelencoder.fit_transform(self.data_clinical_patient['OS_STATUS'])
        self.data_clinical_patient['CLAUDIN_SUBTYPE'] = labelencoder.fit_transform(
            self.data_clinical_patient['CLAUDIN_SUBTYPE'])
        self.data_clinical_patient['THREEGENE'] = labelencoder.fit_transform(self.data_clinical_patient['THREEGENE'])
        self.data_clinical_patient['LATERALITY'] = labelencoder.fit_transform(self.data_clinical_patient['LATERALITY'])
        self.data_clinical_patient['RADIO_THERAPY'] = labelencoder.fit_transform(
            self.data_clinical_patient['RADIO_THERAPY'])
        self.data_clinical_patient['HISTOLOGICAL_SUBTYPE'] = labelencoder.fit_transform(
            self.data_clinical_patient['HISTOLOGICAL_SUBTYPE'])
        self.data_clinical_patient['BREAST_SURGERY'] = labelencoder.fit_transform(
            self.data_clinical_patient['BREAST_SURGERY'])

        onehotencoder = OneHotEncoder(categorical_features=[2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17])
        self.data_clinical_patient = onehotencoder.fit_transform(self.data_clinical_patient).toarray()

        # Feature Scaling
        sc = StandardScaler()
        self.clinical_X = sc.fit_transform(self.data_clinical_patient[:, :-1])
        self.clinical_Y = self.data_clinical_patient[:, -1]
        for i, patient in enumerate(self.clinical_patients):
            self.patient_vector_dict_X[patient] = self.clinical_X[i, :]
            self.patient_vector_dict_Y[patient] = self.clinical_Y[i]

    def preprocess_genomic_data_file(self):
        # Data expression median is the file of the expression extent of the certain gene by some kind of normalization
        self.data_expression_median = pd.read_csv("../data/data_expression_median.txt", sep='\t')

        # data CNA is the file of the copy of gene on the dna
        self.data_CNA = pd.read_csv("../data/data_CNA.txt", sep='\t')

        # data mutations is the file of the mutations of gene
        self.data_mutations_mskcc = pd.read_csv("../data/data_mutations_mskcc.txt", sep='\t', skiprows=1, index_col=0)

        all_patient_dict = {}
        # The target of this preprocess is to build the input matrix of features of gene based on each patient.
        # First extract patient from those three tables
        # for data_expression_median
        for i, patient in enumerate(list(self.data_expression_median.columns[2:])):
            if patient not in all_patient_dict.keys():
                all_patient_dict[patient] = i
        # for data_CNA
        for patient in list(self.data_CNA.columns[2:]):
            if patient not in all_patient_dict.keys():
                cc = len(all_patient_dict)
                all_patient_dict[patient] = len(all_patient_dict)
                dd = len(all_patient_dict)
                if cc != dd - 1:
                    print("dict wrong assigned")
                    exit(0)

        # for data_mutations_mskcc
        with open("../data/data_mutations_mskcc.txt", "r") as fin:
            patient_line = fin.readline()
        patient_line.strip("\n")
        patient_mutation = patient_line.split(" ")[2:]
        for patient in patient_mutation:
            if patient not in all_patient_dict.keys():
                cc = len(all_patient_dict)
                all_patient_dict[patient] = len(all_patient_dict)
                dd = len(all_patient_dict)
                if cc != dd - 1:
                    print("dict wrong assigned")
                    exit(0)

        # extract patients who are in all three tables, don't want to learn from fake data.
        i = 0
        for patient in all_patient_dict.keys():
            if patient in list(self.data_expression_median.columns[2:]):
                if patient in list(self.data_CNA.columns[2:]):
                    if patient in patient_mutation:
                        if patient not in self.patient_dict.keys():
                            self.patient_dict[patient] = i
                            i = i + 1

        # combine patients with clinical data
        all_patients = []
        for patient in self.clinical_patients:
            if patient in self.patient_dict.keys():
                all_patients.append(patient)
                self.clinical_X_genomic_existed.append(self.patient_vector_dict_X[patient])
                self.clinical_Y_genomic_existed.append(self.patient_vector_dict_Y[patient])

        self.data_mutations_mskcc["ID"] = self.data_mutations_mskcc[
                                              "Variant_Classification"] + self.data_mutations_mskcc.index

        labelencoder = LabelEncoder()
        self.data_mutations_mskcc["ID"] = labelencoder.fit_transform(self.data_mutations_mskcc["ID"])

        # for each patient build input matrix from all three tables
        for patient in all_patients:
            # deal with data_expression_median:
            data_array = []
            data_array = data_array + list(self.data_expression_median[patient])

            # deal with data_cna:
            data_array = data_array + (list(self.data_CNA[patient]))

            # deal with data_mutations, for each we only need Variant Classification
            size = len(set(self.data_mutations_mskcc["ID"]))
            mutation_rows = self.data_mutations_mskcc[self.data_mutations_mskcc["Tumor_Sample_Barcode"] == patient]
            lst = list(set(mutation_rows["ID"]))
            if len(lst) > 0:
                data_array += self.lst_one_hot(lst, size)
            else:
                data_array += list(np.zeros(size))
            self.genomic_patient_features.append(data_array)
        self.genomic_X = np.array(self.genomic_patient_features)
        # Feature Scaling
        sc = StandardScaler()
        self.genomic_X = sc.fit_transform(self.genomic_X)
        df = pd.DataFrame(self.genomic_X)
        indexs = np.where(np.isnan(df))
        nulllistindex = list(set(indexs[0]))
        self.genomic_X = np.delete(self.genomic_X, nulllistindex, axis=0)
        self.genomic_Y = np.delete(np.array(self.clinical_Y_genomic_existed), nulllistindex, axis=0)
        self.clinical_X = np.delete(np.array(self.clinical_X_genomic_existed), nulllistindex, axis=0)
        self.clinical_Y = np.delete(np.array(self.clinical_Y_genomic_existed), nulllistindex, axis=0)
        print("Generate genomic data matrix and clinical data matrix successfully")


def load_data():
    # preprocessing data
    preprocessor = Preprocessor()
    clinical_X = preprocessor.clinical_X
    clinical_Y = preprocessor.clinical_Y
    genomic_X = preprocessor.genomic_X
    genomic_Y = preprocessor.genomic_Y

    # devide data set into 8:1:1 as train,validate,test set
    Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y = devide(clinical_X, clinical_Y)
    Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = devide(genomic_X, genomic_Y)
    return Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y

if __name__ == '__main__':
    preprocessor = Preprocessor()
