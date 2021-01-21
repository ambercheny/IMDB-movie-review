import itertools

from collections import Counter
import math
import numpy as np
import pandas as pd

import nltk
import re
import glob
import os

# Read all text files from folder "neg"
path =r'D:\amberstuff\1091\ML\個人競賽\training_dataset\neg'
allFiles = glob.glob(path + "/*.txt")
df_neg = pd.DataFrame()  # create empty df
stories = []        
filenames = []
for file_ in allFiles:
    with open(file_, encoding='utf-8') as f:
        textf = " ".join(line.strip() for line in f)   
    stories.append(textf)    
    filenames.append(os.path.basename(file_[:-4])) # extract filename without .txt

df_neg["filename"] = filenames
df_neg["stories"] = stories
df_neg["Label"] = "neg"

df_neg.head()
df_neg.to_csv (r'D:\amberstuff\1091\ML\個人競賽\neg.csv', index = None, header=True) 

"""strip()
string = '  xoxo love xoxo   '

# Leading and trailing whitespaces are removed
print(string.strip())
"""

# Read all text files from folder "pos"
path =r'D:\amberstuff\1091\ML\個人競賽\training_dataset\pos'
allFiles = glob.glob(path + "/*.txt")
df_pos = pd.DataFrame()  # create empty DF
stories = []        
filenames = []
for file_ in allFiles:
    with open(file_, encoding='utf-8') as f:
        textf = " ".join(line.strip() for line in f)   
    stories.append(textf)    
    filenames.append(os.path.basename(file_[:-4]))    # extract filename without .txt

df_pos["filename"] = filenames
df_pos["stories"] = stories
df_pos["Label"] = "pos"

df_pos.head()
df_pos.to_csv (r'D:\amberstuff\1091\ML\個人競賽\pos.csv', index = None, header=True)


# concatenate df_neg and df_pos and do the following preprocessing work
df_all = pd.concat([df_neg, df_pos])
df_all.to_csv (r'D:\amberstuff\1091\ML\個人競賽\all.csv', index = None, header=True)