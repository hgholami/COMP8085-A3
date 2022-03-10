import sys
import pandas as pd
import numpy as np
import pickle
#from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

args = sys.argv
num_of_args = 1

c_data = data = train_data = test_data = train_target = test_target = None

def read():
    global c_data
    c_data = pd.read_csv('dataset.csv')

    #print(c_data.head())
    #print(c_data.tail())

#def preprocess():


def split():
    global c_data, data, train_data, test_data, train_target, test_target
    data = c_data.iloc[: , 1:]
    target = c_data.iloc[: , 0]

    #print(data.head())
    #print(target.head())
    train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=0.9, random_state=10)

def main():
    if(len(args) != num_of_args):
        print("Expected number of arguments is ", num_of_args, "...")
        return
    
    read()
    #preprocess()
    split()
    #arg1 = args[1]


if __name__ == '__main__':
    main()
