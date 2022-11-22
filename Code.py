import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import os
import sys
import glob
from sklearn.utils import shuffle
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from bokeh.models import HoverTool
from bokeh.plotting import output_notebook, output_file, figure, show, ColumnDataSource
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import os
import sys
import glob
from sklearn.utils import shuffle
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)
from sklearn import metrics

df = pd.read_csv('KDD99.csv', low_memory=False)
print ('Data Loaded')

from sklearn.preprocessing import LabelEncoder
lbl=LabelEncoder()

for col in df.columns:
    df[col] = lbl.fit_transform(df[col])
	
#separate the x and y variables
Y = df['class']
X = df.iloc[:,1:23]

print(df.groupby('class').size())

def splitDataset(X2,Y2):
    #array = df3.values
    #X = array[:,0:sel]
    validation_size = 0.20
    random_seed = 100
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X2, Y2, 
                                                                                    test_size=validation_size, 
                                                                                    random_state=random_seed)
    return X_train, X_validation, Y_train, Y_validation

X_train, X_validation, Y_train, Y_validation = splitDataset(X,Y)

def runBaseExperiments(X_train, Y_train,X_validation,Y_validation):
    random_seed = 100
    scoring = 'accuracy'
    models = []
    GaussianNB(var_smoothing = 0.0992)
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='auto')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    models.append(('AB', AdaBoostClassifier(n_estimators=100)))
    models.append(('DT', DecisionTreeClassifier()))
    models.append(('RF', RandomForestClassifier(n_estimators=10)))

    # evaluate each model in turn using 10-fold cross-validation
    results = []
    classifiers = []
    accuracies = []
	aucs= []
    expected = Y_validation
	
    for classifier, model in models:
        st_time = dt.datetime.now()
        model.fit(X_train, Y_train)
        ed_time = dt.datetime.now()
        time_fit = ed_time-st_time
        predicted = model.predict(X_validation)
        pr_time   = dt.datetime.now()
        time_predict = pr_time-ed_time
		
        accuracy_sv = accuracy_score(expected, predicted)
        recall_sv = recall_score(expected, predicted,average="weighted")
        precision_sv = precision_score(expected, predicted,average="weighted" )
        f1_sv = f1_score(expected, predicted,average="weighted")
        cm_sv = metrics.confusion_matrix(expected, predicted)
		
        accuracies.extend([classifier, accuracy_sv])
		
        results.extend([classifier,time_predict])


		
        classifiers.append([classifier,cm_sv])
        print('Confusion Matrix Heatmap: ' + classifier)
        confusion_matrix_from_sv_df = pd.DataFrame(cm_sv, index = ['Edible','Poisonous'], columns = ['Edible','Poisonous'])
        plt.figure(figsize=(15,10))
        title = 'Confusion Matrix for Classifier on Mushroom :'+ classifier
        plt.title(title)
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        sns.heatmap(confusion_matrix_from_sv_df, annot=True, fmt='g',annot_kws={"size": 10}) # font size
        name = 'CM_' + classifier +'.png'
        plt.savefig(name, dpi=300)
        plt.show()
        
        from sklearn.metrics import roc_curve, roc_auc_score, auc
        fpr, tpr, thresholds = roc_curve(Y_validation, predicted)
        auc = auc(fpr,tpr)
        plt.plot(fpr,tpr, label = 'AUC: %0.2f' % auc)
        plt.plot([0,1], [0,1], 'r--')
        plt.legend()
        name = 'ROC_' + classifier +'.png'
        plt.savefig(name, dpi=300)
        plt.show()
		
		aucs.extend([classifier,fpr,tpr])

       # msg = "%s: %f (classifier, accuracy_sv)
       # print(msg)
    return results, accuracies, classifiers,aucs


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
print("Scaling... ")
trainX = sc.fit_transform(X)
print("Scaled... ")
print("Transformed... ")

# Full
X_train, X_validation, Y_train, Y_validation = splitDataset(trainX,Y)
import datetime as dt
results, accuracies, classifiers = runBaseExperiments(X_train, Y_train,X_validation,Y_validation)
print ('Time')
print(results)
print ('Accuracy')
print (accuracies)
#Full End

#ANOVA
from sklearn.feature_selection import SelectKBest, f_classif, chi2
fs = SelectKBest(f_classif, k=5).fit_transform(trainX,Y)
print(fs)

MI_score = f_classif(trainX, Y, random_state=0)
feature_names = df.columns
k=[]
for feature in zip(feature_names, MI_score):
    print(feature)
    k.append(feature)

X_train, X_validation, Y_train, Y_validation = splitDataset(fs,Y)
results, accuracies, classifiers = runBaseExperiments(X_train, Y_train,X_validation,Y_validation)
print ('Time')
print(results)
print ('Accuracy')
print (accuracies)
# End


#ANOVA
fs = SelectKBest(f_classif, k=10).fit_transform(trainX,Y)
print(fs)

MI_score = f_classif(trainX, Y, random_state=0)
feature_names = df.columns
k=[]
for feature in zip(feature_names, MI_score):
    print(feature)
    k.append(feature)

X_train, X_validation, Y_train, Y_validation = splitDataset(fs,Y)
results, accuracies, classifiers = runBaseExperiments(X_train, Y_train,X_validation,Y_validation)
print ('Time')
print(results)
print ('Accuracy')
print (accuracies)
# End

#ANOVA
fs = SelectKBest(f_classif, k=15).fit_transform(trainX,Y)
print(fs)

MI_score = f_classif(trainX, Y, random_state=0)
feature_names = df.columns
k=[]
for feature in zip(feature_names, MI_score):
    print(feature)
    k.append(feature)

X_train, X_validation, Y_train, Y_validation = splitDataset(fs,Y)
results, accuracies, classifiers = runBaseExperiments(X_train, Y_train,X_validation,Y_validation)
print ('Time')
print(results)
print ('Accuracy')
print (accuracies)
# End

#ANOVA
fs = SelectKBest(f_classif, k=20).fit_transform(trainX,Y)
print(fs)

MI_score = f_classif(trainX, Y, random_state=0)
feature_names = df.columns
k=[]
for feature in zip(feature_names, MI_score):
    print(feature)
    k.append(feature)

X_train, X_validation, Y_train, Y_validation = splitDataset(fs,Y)
results, accuracies, classifiers = runBaseExperiments(X_train, Y_train,X_validation,Y_validation)
print ('Time')
print(results)
print ('Accuracy')
print (accuracies)
# End

#CHI2
fs = SelectKBest(chi2, k=5).fit_transform(trainX,Y)
print(fs)

MI_score = chi2(trainX, Y, random_state=0)
feature_names = df.columns
k=[]
for feature in zip(feature_names, MI_score):
    print(feature)
    k.append(feature)

X_train, X_validation, Y_train, Y_validation = splitDataset(fs,Y)
results, accuracies, classifiers = runBaseExperiments(X_train, Y_train,X_validation,Y_validation)
print ('Time')
print(results)
print ('Accuracy')
print (accuracies)
# End

#CHI2
fs = SelectKBest(chi2, k=10).fit_transform(trainX,Y)
print(fs)

MI_score = chi2(trainX, Y, random_state=0)
feature_names = df.columns
k=[]
for feature in zip(feature_names, MI_score):
    print(feature)
    k.append(feature)

X_train, X_validation, Y_train, Y_validation = splitDataset(fs,Y)
results, accuracies, classifiers = runBaseExperiments(X_train, Y_train,X_validation,Y_validation)
print ('Time')
print(results)
print ('Accuracy')
print (accuracies)
# End

#CHI2
fs = SelectKBest(chi2, k=15).fit_transform(trainX,Y)
print(fs)

MI_score = chi2(trainX, Y, random_state=0)
feature_names = df.columns
k=[]
for feature in zip(feature_names, MI_score):
    print(feature)
    k.append(feature)

X_train, X_validation, Y_train, Y_validation = splitDataset(fs,Y)
results, accuracies, classifiers = runBaseExperiments(X_train, Y_train,X_validation,Y_validation)
print ('Time')
print(results)
print ('Accuracy')
print (accuracies)
# End

#CHI2
fs = SelectKBest(chi2, k=20).fit_transform(trainX,Y)
print(fs)

MI_score = chi2(trainX, Y, random_state=0)
feature_names = df.columns
k=[]
for feature in zip(feature_names, MI_score):
    print(feature)
    k.append(feature)

X_train, X_validation, Y_train, Y_validation = splitDataset(fs,Y)
results, accuracies, classifiers = runBaseExperiments(X_train, Y_train,X_validation,Y_validation)
print ('Time')
print(results)
print ('Accuracy')
print (accuracies)
# End



