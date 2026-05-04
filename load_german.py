import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def remove_A(khar):
    return khar[-1]

def remove_A_purpose(khar):
    if khar != "A410":
        return khar[-1]
    return "10"
    
def age_binary(khar):
    if khar >= 25:
        return 'older'
    if khar < 25:
        return 'younger'

def label_binary(khar):
    if khar == 2:
        return 0
    if khar == 1:
        return 1
           
class GermanDataSet(Dataset):
    
    def __init__(self, data):
        df =  pd.read_csv(data, sep=" ")
        df.columns=['Status of existing checking account','Duration in month','Credit history','Purpose','Credit amount','Savings account/bonds','Present employment since','Installment rate in percentage of disposable income','Personal status and sex','Other debtors / guarantors','Present residence since','Property','Age in years','Other installment plans','Housing','Number of existing credits at this bank','Job','Number of people being liable to provide maintenance for','Telephone','foreign worker','Label']
        new_row=pd.DataFrame({'Status of existing checking account': 'A11','Duration in month':6,'Credit history':'A34','Purpose':'A43','Credit amount':1169,'Savings account/bonds':'A65','Present employment since':'A75','Installment rate in percentage of disposable income':4,'Personal status and sex':'A93','Other debtors / guarantors':'A101','Present residence since':4,'Property':'A121','Age in years':67,'Other installment plans':'A143','Housing':'A152','Number of existing credits at this bank':2,'Job':'A173','Number of people being liable to provide maintenance for':1,'Telephone':'A192','foreign worker':'A201','Label':1},index =[0])
        df= pd.concat([new_row, df]).reset_index(drop = True)
        df['Label']= df['Label'].apply(label_binary)
        df['foreign worker'] = df['foreign worker'].apply(remove_A)
        df['Status of existing checking account'] = df['Status of existing checking account'].apply(remove_A)
        df['Savings account/bonds'] = df['Savings account/bonds'].apply(remove_A)
        df['Credit history'] = df['Credit history'].apply(remove_A)
        df['Other installment plans'] = df['Other installment plans'].apply(remove_A)
        df['Housing'] = df['Housing'].apply(remove_A)
        df['Job'] = df['Job'].apply(remove_A)
        df['Telephone'] = df['Telephone'].apply(remove_A)
        df['Other debtors / guarantors'] = df['Other debtors / guarantors'].apply(remove_A)
        df['Personal status and sex'] = df['Personal status and sex'].apply(remove_A)
        df['Present employment since'] = df['Present employment since'].apply(remove_A)
        df['Property'] = df['Property'].apply(remove_A)
        df['Purpose'] = df['Purpose'].apply(remove_A_purpose)
        df['Age in years'] = df['Age in years'].apply(age_binary)
        df_train_processed = pd.DataFrame()
        for column in df.columns:
            if column in ['Label','Duration in month', 'Credit amount','Installment rate in percentage of disposable income','Present residence since', 'Number of existing credits at this bank','Number of people being liable to provide maintenance for']:
                df_train_processed[column] = df[column]
            else:
                dummies = pd.get_dummies(df[column], prefix=column, dtype=float)
                df_train_processed = pd.concat([df_train_processed, dummies], axis=1)
        df = df_train_processed
        ##############
        #self.columns_after_preprocessing = df_raw.columns.tolist()
        df=df.drop(columns=['Age in years_older'])
        s = df['Age in years_younger']
        scaler = MinMaxScaler()
        y=df['Label']
        X = df.drop(['Label'], axis=1)
        
        
        X = scaler.fit_transform(X)
        self.s = np.array(s)
        self.X = np.array(X)
        self.y = np.array(y)

    def __len__(self):
        return len(self.y)

    def get_all(self):
        return self.X, self.y
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.s[idx]
