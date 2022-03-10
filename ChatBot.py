import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

args = sys.argv
num_of_args = 1

def main():
    if(len(args) != num_of_args):
        print("Expected number of arguments is ", num_of_args, "...")
        return
    
    #arg1 = args[1]


if __name__ == '__main__':
    main()
