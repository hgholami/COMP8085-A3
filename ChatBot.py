import sys, getopt
import pandas as pd
import math
import re
from starter import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

args = sys.argv
num_of_args = 1

c_data = data = train_data = test_data = train_target = test_target = None
symp_desc = symp_prec = symp_sev = None
t_data = None
use_t_data = False
p_disease = p_has_symptom_disease = None
diseases = prob_diseases = dict()
symptoms_of_diseases_dict = dict(defaultdict=set)

def read(t_data_path=None):
    global c_data, symp_desc, symp_prec, symp_sev, t_data, use_t_data
    c_data = pd.read_csv('dataset.csv')
    for col in c_data.iloc[:,1:].columns:
        if pd.api.types.is_string_dtype(c_data[col]):
            c_data[col] = c_data[col].str.strip()
            c_data[col] = c_data[col].str.replace("\s*[ ]\s*", "", regex=True)
    c_data = c_data.fillna('None')
    
    symp_desc = pd.read_csv('symptom_Description.csv', header=None)

    symp_prec = pd.read_csv('symptom_precaution.csv', header=None)
    symp_prec = symp_prec.fillna('None')

    symp_sev = pd.read_csv('Symptom_severity.csv', header=None)

    #If testdata given, read and save as DataFrame
    if t_data_path != None:
        use_t_data = True
        t_data = pd.read_csv(t_data_path)
        for col in t_data.iloc[:,1:].columns:
            if pd.api.types.is_string_dtype(t_data[col]):
                t_data[col] = t_data[col].str.strip()
                t_data[col] = t_data[col].str.replace("\s*[ ]\s*", "", regex=True)
        t_data = t_data.fillna('None')

def split():
    global c_data, data, train_data, test_data, train_target, test_target
    data = c_data.iloc[: , 1:]
    target = c_data.iloc[: , 0]

    if not use_t_data:
        train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=0.9, random_state=42)
    else:
        train_data = data
        train_target = target
        test_data = t_data.iloc[:,1:]
        test_target = t_data.iloc[:,0]

def train():
    global c_data, data, train_data, test_data, train_target, test_target, p_disease, p_has_symptom_disease, diseases, prob_diseases

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    train_target = train_target.reset_index(drop=True)
    test_target = test_target.reset_index(drop=True)
    
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

    
def validate():
    global test_data, test_target, p_disease, p_has_symptom_disease, diseases, prob_diseases

    prediction_list = list()
    
    #Within test data
    for i in range(0, len(test_target)):
        symptoms = test_data.iloc[i,:]

        clean_symptoms = [symp for symp in symptoms if symp != 'None']
        for key in prob_diseases:
            prob_diseases[key] = p_disease[key] * math.prod((p_has_symptom_disease[key,symp] for symp in clean_symptoms))

        prediction_list.append(max(prob_diseases, key=prob_diseases.get))
        
        prob_diseases = prob_diseases.fromkeys(prob_diseases, 0)
        
    print(classification_report(test_target.values, prediction_list))

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

    while not valid:
        print("I found",len(symps_found),"symptom names matching your symptom.",
              "\nPlease confirm your intended option:")
        
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

    for key in prob_diseases:
            prob_diseases[key] = p_disease[key] * p_has_symptom_disease[key,symp]
    
    probable_diseases = list()
    probable_symptoms = list()
    for key in prob_diseases:
        if prob_diseases[key] > 0:
            probable_diseases.append(key)


    for probd in probable_diseases:
        probable_symptoms.extend([x for x in symptoms_of_diseases_dict[probd] if x not in probable_symptoms])
    
    
    if len(probable_diseases) > 1: 
        print("I see, I have a hypothesis, let me test it further,")
        print("Please answer \"yes\" or \"no\" to the following questions:")

    confirmed_symptoms = [symp]
    probable_symptoms.remove(symp)

    while len(probable_diseases) > 1 and len(probable_symptoms) > 1:
        answer = input("Are you experiencing " + str(probable_symptoms[0]) + "?\nUser Input: ")
        if str.lower(answer).strip() == "yes":
            confirmed_symptoms.append(probable_symptoms[0])
            probable_symptoms.remove(probable_symptoms[0])

            for key in probable_diseases:
                prob_diseases[key] = p_disease[key] * math.prod((p_has_symptom_disease[key, c_symp] for c_symp in confirmed_symptoms))
            
            for key in prob_diseases:
                if prob_diseases[key] <= 0 and key in probable_diseases:
                    probable_diseases.remove(str(key))

        elif str.lower(answer).strip() == "no":
            probable_symptoms.remove(str(probable_symptoms[0]))
        else:
            print("Please enter answer \"yes\" or \"no\"...")
    
    if(len(probable_diseases) == 0):
        print("No known diseases are probable based on symptoms given. Exiting...")
        return

    most_probable_disease = probable_diseases[probable_diseases.index(max(prob_diseases, key=prob_diseases.get))]

    sev_sum = 0
    for c_sym in confirmed_symptoms:
        sev_sum += int(symp_sev.loc[symp_sev[0] == c_sym][1])

    sick_sev_score = (sev_sum*dur)/(len(confirmed_symptoms)+1)

    if sick_sev_score > 13:
        print("You should seek consultation from a doctor!")
    else:
        print("It might not be that bad but you should take precautions.")
        prob_sum = sum(prob_diseases[key] for key in probable_diseases)
        prob_score = round(prob_diseases[most_probable_disease]/prob_sum * 100, 2)
        print("I'm " + str(prob_score) + "% confident you are facing",most_probable_disease + ".")
        print("Take following measures:")
        
        
        columns = list(symp_prec[symp_prec.iloc[:,0] == most_probable_disease].iloc[:,1:])
        for i in columns:
            print ("    ",i,")",symp_prec[symp_prec.iloc[:,0] == most_probable_disease].iloc[:,1:][i].values[0])


def main(argv):
    testdatapath = None
    bot_opt = valid_opt = False
    try:
        opts, args = getopt.getopt(argv,"hd:bv", ["data=", "bot", "validate"])
    except getopt.GetoptError:
        print('python ChatBot.py -d <testdata.csv>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('ChatBot.py #Will train and test bot with given dataset.csv',
            '\n-d OR --data <testdata.csv> #[Optional]Will train on dataset.csv and test using <testdata.csv>',
            '\n-b OR --bot #Will run chatbot',
            '\n-v OR --validate #Will show validation scores')
            sys.exit()
        elif opt in ("-d", "--data"):
            testdatapath = arg
            valid_opt = True
        elif opt in ("-b", "--bot"):
            bot_opt = True
        elif opt in ("-v", "--validate"):
            valid_opt = True

            
    read(testdatapath)
    split()
    train()
    if valid_opt: validate()
    if bot_opt: run_bot()
    
    

if __name__ == '__main__':
    main(sys.argv[1:])
