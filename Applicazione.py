import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#LOAD THE MODEL
filename = '/Users/valeninaberardi/Desktop/joblib_model.sav'
import joblib
model = joblib.load(filename)
#ASK FOR THE FILE TO ANALYZE
fileinput = input("Insert the file to analyze: ")
#READ THE FILE
rowToAnalyze = pd.read_csv(fileinput)
#PREDICT
prediction = model.predict([rowToAnalyze.iloc[0]])
#SHOW THE RESULT
print("The gravity of the car accident is: " + prediction[0].astype(str))