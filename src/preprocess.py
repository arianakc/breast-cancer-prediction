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
        self.patient_dict = {}
        self.genomic_patient_features = []
        self.genomic_patient_feature_matrix = None
        # if there is no data files uncomment the following code.
        # self.get_data_file()

        #self.preprocess_clinical_data_file()
        self.preprocess_genomic_data_file()

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

    def preprocess_genomic_data_file(self):
        # Data expression median is the file of the expression extent of the certain gene by some kind of normalization
        self.data_expression_median = pd.read_csv("../data/data_expression_median.txt", sep='\t')

        # data CNA is the file of the copy of gene on the dna
        self.data_CNA = pd.read_csv("../data/data_CNA.txt", sep='\t')

        # data mutations is the file of the mutations of gene
        self.data_mutations_mskcc = pd.read_csv("../data/data_mutations_mskcc.txt", sep='\t', skiprows=1)

        print("extract genomic data successfully!")

        all_patient_dict = {}
        # The target of this preprocess is to build the input matrix of features of gene based on each patient.
        # First extract patient from those three tables
        # for data_expression_median
        for i, patient in enumerate(list(self.data_expression_median.columns[2:])):
            if patient not in  all_patient_dict.keys():
                all_patient_dict[patient] = i
        # for data_CNA
        for patient in list(self.data_CNA.columns[2:]):
            if patient not in  all_patient_dict.keys():
                cc = len( all_patient_dict)
                all_patient_dict[patient] = len( all_patient_dict)
                dd = len( all_patient_dict)
                if cc != dd-1:
                    print("dict wrong assigned")
                    exit(0)

        # for data_mutations_mskcc
        with open("../data/data_mutations_mskcc.txt", "r") as fin:
            patient_line = fin.readline()
        print(patient_line)
        patient_line.strip("\n")
        patient_mutation = patient_line.split(" ")[2:]
        for patient in patient_mutation:
            if patient not in  all_patient_dict.keys():
                cc = len( all_patient_dict)
                all_patient_dict[patient] = len( all_patient_dict)
                dd = len( all_patient_dict)
                if cc != dd-1:
                    print("dict wrong assigned")
                    exit(0)

        print("all_patient_dict:",all_patient_dict)

        # extract patients who are in all three tables, don't want to learn from fake data.
        i = 0
        for patient in all_patient_dict.keys():
            if patient in list(self.data_expression_median.columns[2:]):
                if patient in list(self.data_CNA.columns[2:]):
                    if patient in patient_mutation:
                        if patient not in self.patient_dict.keys():
                            self.patient_dict[patient] = i
                            i = i + 1
        print(self.patient_dict)


        #extract all dna name in data_mutations:
        mutation_dnas = []
        for dna in self.data_mutations_mskcc["Hugo_Symbol"]:
            if dna not in mutation_dnas:
                mutation_dnas.append(dna)

        labelencoder = LabelEncoder()
        self.data_mutations_mskcc["Consequence"] = labelencoder.fit_transform(self.data_mutations_mskcc["Consequence"])
        self.data_mutations_mskcc["Variant_Classification"] = labelencoder.fit_transform(self.data_mutations_mskcc["Variant_Classification"])

        # for each patient build input matrix from all three tables
        for patient in self.patient_dict.keys():
            # deal with data_expression_median:
            data_array = []
            data_array = data_array+list(self.data_expression_median[patient])

            # deal with data_cna:
            data_array = data_array + (list(self.data_CNA[patient]))

            # deal with data_mutations, for each we only need Variant Classification and Consequence
            mutation_data_array = []
            if patient in self.data_mutations_mskcc["Tumor_Sample_Barcode"]:
                data_mutation_rows = self.data_mutations_mskcc.loc[self.data_mutations_mskcc["Tumor_Sample_Barcode"] == patient]
                for dna in mutation_dnas:
                    if dna in data_mutation_rows["Hugo_Symbol"]:
                        mutation_data_array.append(data_mutation_rows["Consequence"])
                        mutation_data_array.append(data_mutation_rows["Variant_Classification"])
                    else:
                        mutation_data_array.append(-1)
                        mutation_data_array.append(-1)
            else:
                for dna in mutation_dnas:
                    mutation_data_array.append(-1)
                    mutation_data_array.append(-1)
            data_array = data_array + mutation_data_array
            self.genomic_patient_features.append(data_array)
        self.genomic_patient_feature_matrix = np.array(self.genomic_patient_features)
        print("Generate clinical data matrix successfully without one-hot encoding and feature scaling")


if __name__ == '__main__':
    preprocessor = Preprocessor()
