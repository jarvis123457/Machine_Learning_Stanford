# Packages

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

%matplotlib inline

# Logistic Regression

# load dataset
X_train, y_train = load_data("data/ex2data1.txt")
print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))
