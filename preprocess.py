import numpy as np
from util import *

random_seed = 0

np.random.seed(random_seed)

# do train/validate/test split, we want to do 60%/20%/20%
from sklearn.model_selection import train_test_split
import pandas as pd
import os

os.mkdir('preprocess')
os.mkdir('preprocess/train')
os.mkdir('preprocess/test')
os.mkdir('preprocess/validate')

df = pd.read_csv('train.csv', index_col=0).sample(frac=1.0)

n_sample = len(df)
n_train = int(0.6 * n_sample)
n_validate = int(0.2 * n_sample)
n_test = n_sample - n_train - n_validate

df['type'] = ['train'] * n_train + ['validate'] * n_validate + ['test'] * n_test

def save_csv(type):
  data = df[df['type'] == type].copy().drop('type', axis=1)
  data.to_csv('preprocess/{0}/{0}.csv'.format(type))

save_csv('train')
save_csv('test')
save_csv('validate')