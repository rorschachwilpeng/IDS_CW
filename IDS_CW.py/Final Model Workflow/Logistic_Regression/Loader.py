import xlrd
import pickle
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
####################################################################
####################### Obtain 2018 Test Data #####################
####################################################################
# --------------------------------------------------
# Obtain the testging data (Q1)
# --------------------------------------------------

dataQ12018 = xlrd.open_workbook("../../Test Datasets/2018/data_Q12018.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ12018 = dataQ12018.sheet_by_name('Q12018')

# 获取工作表的名称，行数，列数
nameQ12018 = tableQ12018.name
rowNumQ12018 = tableQ12018.nrows
colNumQ12018 = tableQ12018.ncols

# 获取每个column的信息
CS_test_Q12018 = [float(tableQ12018.cell_value(i, 1)) for i in range(1, tableQ12018.nrows)]
LTV_test_Q12018 = [float(tableQ12018.cell_value(i, 2)) for i in range(1, tableQ12018.nrows)]
DTI_test_Q12018 = [float(tableQ12018.cell_value(i, 3)) for i in range(1, tableQ12018.nrows)]
UPB_test_Q12018 = [float(tableQ12018.cell_value(i, 4)) for i in range(1, tableQ12018.nrows)]
DEFAULT_test_Q12018 = [float(tableQ12018.cell_value(i, 5)) for i in range(1, tableQ12018.nrows)]
IR_test_Q12018 = [float(tableQ12018.cell_value(i, 6)) for i in range(1, tableQ12018.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_test_Q12018 = np.array(CS_test_Q12018,dtype=float)
CS_test_Q12018.resize((261690, 1))

LTV_test_Q12018 = np.array(LTV_test_Q12018,dtype=float)
LTV_test_Q12018.resize((261690, 1))

DTI_test_Q12018 = np.array(DTI_test_Q12018,dtype=float)
DTI_test_Q12018.resize((261690, 1))

UPB_test_Q12018 = np.array(UPB_test_Q12018,dtype=float)
UPB_test_Q12018.resize((261690, 1))

IR_test_Q12018 = np.array(IR_test_Q12018,dtype=float)
IR_test_Q12018.resize((261690, 1))

x_test_Q12018 = np.concatenate((CS_test_Q12018,LTV_test_Q12018),axis=1)
x_test_Q12018 = np.concatenate((x_test_Q12018,DTI_test_Q12018),axis=1)
x_test_Q12018 = np.concatenate((x_test_Q12018,UPB_test_Q12018),axis=1)
x_test_Q12018 = np.concatenate((x_test_Q12018,IR_test_Q12018),axis=1)

# Reshape and turn the list into arr(y)
y_test_Q12018 = np.array(DEFAULT_test_Q12018,dtype=int)
y_test_Q12018.resize((261690,))



# --------------------------------------------------
# Obtain the testging data (Q2)
# --------------------------------------------------
dataQ22018 = xlrd.open_workbook("../../Test Datasets/2018/data_Q22018.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ22018 = dataQ22018.sheet_by_name('Q22018')

# 获取工作表的名称，行数，列数
nameQ22018 = tableQ22018.name
rowNumQ22018 = tableQ22018.nrows
colNumQ22018 = tableQ22018.ncols

# 获取每个column的信息
CS_test_Q22018 = [float(tableQ22018.cell_value(i, 1)) for i in range(1, tableQ22018.nrows)]
LTV_test_Q22018 = [float(tableQ22018.cell_value(i, 2)) for i in range(1, tableQ22018.nrows)]
DTI_test_Q22018 = [float(tableQ22018.cell_value(i, 3)) for i in range(1, tableQ22018.nrows)]
UPB_test_Q22018 = [float(tableQ22018.cell_value(i, 4)) for i in range(1, tableQ22018.nrows)]
DEFAULT_test_Q22018 = [float(tableQ22018.cell_value(i, 5)) for i in range(1, tableQ22018.nrows)]
IR_test_Q22018 = [float(tableQ22018.cell_value(i, 6)) for i in range(1, tableQ22018.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_test_Q22018 = np.array(CS_test_Q22018,dtype=float)
CS_test_Q22018.resize((305067, 1))

LTV_test_Q22018 = np.array(LTV_test_Q22018,dtype=float)
LTV_test_Q22018.resize((305067, 1))

DTI_test_Q22018 = np.array(DTI_test_Q22018,dtype=float)
DTI_test_Q22018.resize((305067, 1))

UPB_test_Q22018 = np.array(UPB_test_Q22018,dtype=float)
UPB_test_Q22018.resize((305067, 1))

IR_test_Q22018 = np.array(IR_test_Q22018,dtype=float)
IR_test_Q22018.resize((305067, 1))

x_test_Q22018 = np.concatenate((CS_test_Q22018,LTV_test_Q22018),axis=1)
x_test_Q22018 = np.concatenate((x_test_Q22018,DTI_test_Q22018),axis=1)
x_test_Q22018 = np.concatenate((x_test_Q22018,UPB_test_Q22018),axis=1)
x_test_Q22018 = np.concatenate((x_test_Q22018,IR_test_Q22018),axis=1)

# Reshape and turn the list into arr(y)
y_test_Q22018 = np.array(DEFAULT_test_Q22018,dtype=int)
y_test_Q22018.resize((305067,))

# --------------------------------------------------
# Obtain the testging data (Q3)
# --------------------------------------------------
dataQ32018 = xlrd.open_workbook("../../Test Datasets/2018/data_Q32018.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ32018 = dataQ32018.sheet_by_name('Q32018')

# 获取工作表的名称，行数，列数
nameQ32018 = tableQ32018.name
rowNumQ32018 = tableQ32018.nrows
colNumQ32018 = tableQ32018.ncols

# 获取每个column的信息
CS_test_Q32018 = [float(tableQ32018.cell_value(i, 1)) for i in range(1, tableQ32018.nrows)]
LTV_test_Q32018 = [float(tableQ32018.cell_value(i, 2)) for i in range(1, tableQ32018.nrows)]
DTI_test_Q32018 = [float(tableQ32018.cell_value(i, 3)) for i in range(1, tableQ32018.nrows)]
UPB_test_Q32018 = [float(tableQ32018.cell_value(i, 4)) for i in range(1, tableQ32018.nrows)]
DEFAULT_test_Q32018 = [float(tableQ32018.cell_value(i, 5)) for i in range(1, tableQ32018.nrows)]
IR_test_Q32018 = [float(tableQ32018.cell_value(i, 6)) for i in range(1, tableQ32018.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_test_Q32018 = np.array(CS_test_Q32018,dtype=float)
CS_test_Q32018.resize((279430, 1))

LTV_test_Q32018 = np.array(LTV_test_Q32018,dtype=float)
LTV_test_Q32018.resize((279430, 1))

DTI_test_Q32018 = np.array(DTI_test_Q32018,dtype=float)
DTI_test_Q32018.resize((279430, 1))

UPB_test_Q32018 = np.array(UPB_test_Q32018,dtype=float)
UPB_test_Q32018.resize((279430, 1))

IR_test_Q32018 = np.array(IR_test_Q32018,dtype=float)
IR_test_Q32018.resize((279430, 1))

x_test_Q32018 = np.concatenate((CS_test_Q32018,LTV_test_Q32018),axis=1)
x_test_Q32018 = np.concatenate((x_test_Q32018,DTI_test_Q32018),axis=1)
x_test_Q32018 = np.concatenate((x_test_Q32018,UPB_test_Q32018),axis=1)
x_test_Q32018 = np.concatenate((x_test_Q32018,IR_test_Q32018),axis=1)

# Reshape and turn the list into arr(y)
y_test_Q32018 = np.array(DEFAULT_test_Q32018,dtype=int)
y_test_Q32018.resize((279430,))

# --------------------------------------------------
# Obtain the testging data (Q4)
# --------------------------------------------------
dataQ42018 = xlrd.open_workbook("../../Test Datasets/2018/data_Q42018.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ42018 = dataQ42018.sheet_by_name('Q42018')

# 获取工作表的名称，行数，列数
nameQ42018 = tableQ42018.name
rowNumQ42018 = tableQ42018.nrows
colNumQ42018 = tableQ42018.ncols

# 获取每个column的信息
CS_test_Q42018 = [float(tableQ42018.cell_value(i, 1)) for i in range(1, tableQ42018.nrows)]
LTV_test_Q42018 = [float(tableQ42018.cell_value(i, 2)) for i in range(1, tableQ42018.nrows)]
DTI_test_Q42018 = [float(tableQ42018.cell_value(i, 3)) for i in range(1, tableQ42018.nrows)]
UPB_test_Q42018 = [float(tableQ42018.cell_value(i, 4)) for i in range(1, tableQ42018.nrows)]
DEFAULT_test_Q42018 = [float(tableQ42018.cell_value(i, 5)) for i in range(1, tableQ42018.nrows)]
IR_test_Q42018 = [float(tableQ42018.cell_value(i, 6)) for i in range(1, tableQ42018.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_test_Q42018 = np.array(CS_test_Q42018,dtype=float)
CS_test_Q42018.resize((244566, 1))

LTV_test_Q42018 = np.array(LTV_test_Q42018,dtype=float)
LTV_test_Q42018.resize((244566, 1))

DTI_test_Q42018 = np.array(DTI_test_Q42018,dtype=float)
DTI_test_Q42018.resize((244566, 1))

UPB_test_Q42018 = np.array(UPB_test_Q42018,dtype=float)
UPB_test_Q42018.resize((244566, 1))

IR_test_Q42018 = np.array(IR_test_Q42018,dtype=float)
IR_test_Q42018.resize((244566, 1))

x_test_Q42018 = np.concatenate((CS_test_Q42018,LTV_test_Q42018),axis=1)
x_test_Q42018 = np.concatenate((x_test_Q42018,DTI_test_Q42018),axis=1)
x_test_Q42018 = np.concatenate((x_test_Q42018,UPB_test_Q42018),axis=1)
x_test_Q42018 = np.concatenate((x_test_Q42018,IR_test_Q42018),axis=1)

# Reshape and turn the list into arr(y)
y_test_Q42018 = np.array(DEFAULT_test_Q42018,dtype=int)
y_test_Q42018.resize((244566,))

# Combine the 2018 testing dataset
x_test_2018 = np.concatenate((x_test_Q12018,x_test_Q22018,x_test_Q32018,x_test_Q42018),axis=0)
y_test_2018 = np.concatenate((y_test_Q12018,y_test_Q22018,y_test_Q32018,y_test_Q42018),axis=0)



# load the model from disk
filename = 'Logistic Regession Trained Model.sav'
model_balanced = pickle.load(open(filename, 'rb'))
# predict the test result
predict_test = model_balanced.predict(x_test_2018)
# test_acc = round(accuracy_score(y_test_2018,predict_test),4)
# Confusion Matrix:
import pandas as pd
result_test = pd.DataFrame(confusion_matrix(y_test_2018,predict_test),index=[0,1],columns=[0,1])
print('confusion matrix of test dataset:\n',result_test)
# Write the confusion matrix as excel file
result_test.to_excel('../OUTPUT/LR_confusion_matrix.xls')

# The testing accurancy
# predict_test = model_balanced.predict(x_test)
# test_acc = round(accuracy_score(y_test,predict_test),4)

# # Confusion Matrix:
# import pandas as pd
# result_test = pd.DataFrame(confusion_matrix(y_test,predict_test),index=[0,1],columns=[0,1])
# print('confusion matrix of test dataset:\n',result_test)
# result_test.to_excel('../OUTPUT/LR_confusion_matrix_test.xls')

# # Output
# print('test accurancy = ',test_acc, '\n', 'Test accurancy = ',test_acc,'\n')




import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np
from sklearn.linear_model import LogisticRegression
# Testing DATASET ROC-AUC DIAGRAM
predictions_test = model_balanced.predict_proba(x_test_2018)
fpr_test,tpr_test,threshold_test = roc_curve(y_test_2018,predictions_test[:,1])
roc_auc_test = auc(fpr_test,tpr_test)
print('The Testing Set AUC: ',roc_auc_test)
# fpr,tpr,threshold = roc_curve(y_test,predictions)
# roc_auc = auc(y_test, predictions)
plt.plot(fpr_test, tpr_test, label='ROC curve (area = %0.3f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.plot([0, 0.5], [1, 0.5], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic of Logistic Regression(Test Set)')
plt.legend(loc="lower right")
plt.savefig('../OUTPUT/LR_ROC-AUC Curve(Test).png')
plt.show()