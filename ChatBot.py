from collections import defaultdict
import sys
import pandas as pd
import numpy as np
import math
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
diseases = prob_diseases = dict()
symptoms_of_diseases_dict = dict(defaultdict=set)

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
    train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=0.9, random_state=42)

def train():
    global c_data, data, train_data, test_data, train_target, test_target, p_disease, p_has_symptom_disease, diseases, prob_diseases

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    train_target = train_target.reset_index(drop=True)
    test_target = test_target.reset_index(drop=True)

    # diseases = dict()
    # prob_diseases = dict()
    
    for disease in train_target:
        if(disease not in diseases):
            diseases.update({disease:1})
            prob_diseases.update({disease:1})
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

                if train_target[i] not in symptoms_of_diseases_dict:
                    symptoms_of_diseases_dict.update({train_target[i]:{symptom}})
                else:
                    symptoms_of_diseases_dict[train_target[i]].add(symptom)

    p_has_symptom_disease.normalize()
    #print(p_has_symptom_disease.show_approx())
    # print('Disease dict:', diseases)
    # print('Test target length:',len(test_target))
    

    # print('target;',test_target)

    
def validate():
    global test_data, test_target, p_disease, p_has_symptom_disease, diseases, prob_diseases

    prediction_list = list()
    # print('p_disease:',p_disease.show_approx())
    # print("")
    # print('p_has_symptom_disease', p_has_symptom_disease.show_approx())
    
    #Within 492 test data
    for i in range(0, len(test_target)):
        symptoms = test_data.iloc[i,:]

        clean_symptoms = [symp for symp in symptoms if symp != 'None']
        # print(clean_symptoms)
        for key in prob_diseases:
            # for symp in clean_symptoms:
            # print(p_has_symptom_disease['Pneumonia',' chills'])
            # print(p_disease[key] * math.prod((p_has_symptom_disease[key,symp] for symp in clean_symptoms)))
            prob_diseases[key] = p_disease[key] * math.prod((p_has_symptom_disease[key,symp] for symp in clean_symptoms))
            # print(key,prob_diseases[key])
            #prob_diseases[key] *= p_has_symptom_disease[key, symptom]
                
        # for key in diseases:
        #     prob_diseases[key] *= p_disease[key]
    
        #print(prob_diseases[max(prob_diseases)])
        prediction_list.append(max(prob_diseases, key=prob_diseases.get))
        # maximum = max(prob_diseases, key=prob_diseases.get)  # Just use 'min' instead of 'max' for minimum.
        # print(maximum, prob_diseases[maximum])
        #print('prob_diseases:',prob_diseases)
        
        prob_diseases = prob_diseases.fromkeys(prob_diseases, 1)
        
    #print('prediction list:',prediction_list)

    #print('test_target:',test_target.values)

    print(p_disease['Acne']*p_has_symptom_disease['Acne','blackheads'])
    print(classification_report(test_target.values, prediction_list))


    # 2d array of disease true positive false negative and total count
    #TODO Figure out how to represent each disease i.e list
    # for i in range(0, len(test_target)):
    #     symptoms = test_data.iloc[i,:]
        
    #     for symptom in symptoms:
    #         if symptom != 'None':
    #             p_has_symptom_disease[str(train_target[i]),str(symptom)]+=1


#def classify():

def run_bot():
    global p_disease, p_has_symptom_disease, diseases, prob_diseases
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
    # print("Patient has been experiencing",symp,"for",dur,"days!")

    for key in prob_diseases:
            prob_diseases[key] = p_disease[key] * p_has_symptom_disease[key,symp]
    
    # print(max(prob_diseases, key=prob_diseases.get))
    # print(prob_diseases)
    print(prob_diseases)
    probable_diseases = list()
    probable_symptoms = list()
    for key in prob_diseases:
        if prob_diseases[key] > 0:
            probable_diseases.append(key)
            #print(key)


    for probd in probable_diseases:
        probable_symptoms.extend([x for x in symptoms_of_diseases_dict[probd] if x not in probable_symptoms])
    
    print("I see, I have a hypothesis, let me test it further,")
    print("Please answer \"yes\" or \"no\" to the following questions:")

    confirmed_symptoms = [symp]
    # print(probable_diseases)
    while len(probable_diseases) > 1 and len(probable_symptoms) > 1:
        answer = input("Are you experiencing " + str(probable_symptoms[0]))
        if str.lower(answer).strip() == "yes":
            confirmed_symptoms.append(probable_symptoms[0])
            probable_symptoms.remove(probable_symptoms[0])

            for key in probable_diseases:
                prob_diseases[key] = p_disease[key] * math.prod((p_has_symptom_disease[key, c_symp] for c_symp in confirmed_symptoms))
            
            for key in prob_diseases:
                if prob_diseases[key] <= 0:
                    probable_diseases.remove(key)
        elif str.lower(answer).strip() == "no":
            probable_symptoms.remove(probable_symptoms[0])
        else:
            print("Please enter answer \"yes\" or \"no\"...")
    
    # print(probable_symptoms)
    # print(probable_diseases)

    # for index, row in (c_data[c_data.iloc[:,0]]).iterrows():
    #     if index in probable_diseases:
    #         print(index, row)
    #         #probable_symptoms.add()

    # print(c_data)
    # print(probable_diseases)
    # for probd in probable_diseases:
    #     for row in (c_data[c_data.iloc[:,0] == probd]):
    #         probable_symptoms.add()
    
    #print(probable_symptoms)


def main():
    if(len(args) != num_of_args):
        print("Expected number of arguments is ", num_of_args, "...")
        return
    
    read()
    #preprocess()
    split()
    train()
    validate()
    #run_bot()

if __name__ == '__main__':
    main()
