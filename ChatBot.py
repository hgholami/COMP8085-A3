import sys
import pandas as pd
import numpy as np
import re
import pickle
from starter import *
#from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

args = sys.argv
num_of_args = 1

c_data = data = train_data = test_data = train_target = test_target = None
symp_desc = symp_prec = symp_sev = None
p_disease = p_has_symptom_disease = None
def read():
    global c_data, symp_desc, symp_prec, symp_sev
    c_data = pd.read_csv('dataset.csv')
    symp_desc = pd.read_csv('symptom_Description.csv', header=None)
    symp_prec = pd.read_csv('symptom_precaution.csv', header=None)
    symp_sev = pd.read_csv('Symptom_severity.csv', header=None)

    #print(c_data.head())
    #print(c_data.tail())

#def preprocess():
    #le = preprocessing.LabelEncoder()
    

def split():
    global c_data, data, train_data, test_data, train_target, test_target
    c_data = c_data.fillna('None')
    data = c_data.iloc[: , 1:]
    target = c_data.iloc[: , 0]
    
    #print(data.head())
    #print(target.head())
    train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=0.9, random_state=10)

def train():
    global c_data, data, train_data, test_data, train_target, test_target, p_disease, p_has_symptom_disease

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    train_target = train_target.reset_index(drop=True)
    test_target = test_target.reset_index(drop=True)

    diseases = dict()

    for disease in train_target:
        if(disease not in diseases):
            diseases.update({disease:1})
        else:
            diseases[disease] += 1

    p_disease = ProbDist("Disease", freqs=diseases)

    variables = ['Disease', 'Symptom']
    p_has_symptom_disease = JointProbDist(variables)

    for i in range(0, len(train_target)):
        symptoms = train_data.iloc[i,:]
        for symptom in symptoms:
            if symptom != 'None':
                p_has_symptom_disease[str(train_target[i]),str(symptom)]+=1

    p_has_symptom_disease.normalize()

    print(p_has_symptom_disease.show_approx())

    # 2d array of disease true positive false negative and total count
    #TODO Figure out how to represent each disease i.e list
    # for i in range(0, len(test_target)):
    #     symptoms = test_data.iloc[i,:]
        
    #     for symptom in symptoms:
    #         if symptom != 'None':
    #             p_has_symptom_disease[str(train_target[i]),str(symptom)]+=1
    


#def classify():

def run_bot():
    print("Hello,","\nPlease tell me about the first symptom you are experiencing...")
    symp = input("User Input: ")
    symp = symp.strip()

    symps_found = symp_sev[symp_sev.iloc[:,0].str.contains(symp, flags=re.IGNORECASE, regex=True)]

    while len(symps_found) <= 0:
        print("I'm sorry, but I'm facing difficulty understanding the symptom.",
              "\nCan you use another word to describe your symptom?")
        symp = input("User Input: ")
        symp = symp.strip()
        symps_found = symp_sev[symp_sev.iloc[:,0].str.contains(symp, flags=re.IGNORECASE, regex=True)]

    symps_found = symps_found.reset_index()

    valid = False
    #while len(symps_found) > 1:
    while not valid:
        print("I found",len(symps_found),"symptom names matching your symptom.",
              "\nPlease confirm your intended option:")
        
        #valid = False
        while not valid:
            for row in symps_found.itertuples():
                print(" ",row[0], ")",row[2])
            symp_con =  int(input("User Input: "))
            if symp_con >= 0 and symp_con < len(symps_found):
                valid = True
            else:
                print("Please enter a number in range:")
        symps_found = symps_found.iloc[[symp_con]]
    
    symp = symps_found.iloc[0][0]

    print("I see, how long have you been experiencing",symp + "?")
    dur = int(input("User Input: "))
    print("Patient has been experiencing",symp,"for",dur,"days!")
            

def main():
    if(len(args) != num_of_args):
        print("Expected number of arguments is ", num_of_args, "...")
        return
    
    read()
    #preprocess()
    split()
    train()
    #run_bot()

if __name__ == '__main__':
    main()
