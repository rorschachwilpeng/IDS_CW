# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import xlrd
####################################################################
####################### Obtain 2015 Train Data #####################
####################################################################
# --------------------------------------------------
# Obtain the trainging data (Q1)
# --------------------------------------------------

dataQ12015 = xlrd.open_workbook("../../../Test Datasets/2015/data_Q12015.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ12015 = dataQ12015.sheet_by_name('Q12015')

# 获取工作表的名称，行数，列数
nameQ12015 = tableQ12015.name
rowNumQ12015 = tableQ12015.nrows
colNumQ12015 = tableQ12015.ncols

# 获取每个column的信息
CS_train_Q12015 = [float(tableQ12015.cell_value(i, 1)) for i in range(1, tableQ12015.nrows)]
LTV_train_Q12015 = [float(tableQ12015.cell_value(i, 2)) for i in range(1, tableQ12015.nrows)]
DTI_train_Q12015 = [float(tableQ12015.cell_value(i, 3)) for i in range(1, tableQ12015.nrows)]
UPB_train_Q12015 = [float(tableQ12015.cell_value(i, 4)) for i in range(1, tableQ12015.nrows)]
DEFAULT_train_Q12015 = [float(tableQ12015.cell_value(i, 5)) for i in range(1, tableQ12015.nrows)]
IR_train_Q12015 = [float(tableQ12015.cell_value(i, 6)) for i in range(1, tableQ12015.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q12015 = np.array(CS_train_Q12015,dtype=float)
CS_train_Q12015.resize((324050, 1))

LTV_train_Q12015 = np.array(LTV_train_Q12015,dtype=float)
LTV_train_Q12015.resize((324050, 1))

DTI_train_Q12015 = np.array(DTI_train_Q12015,dtype=float)
DTI_train_Q12015.resize((324050, 1))

UPB_train_Q12015 = np.array(UPB_train_Q12015,dtype=float)
UPB_train_Q12015.resize((324050, 1))

IR_train_Q12015 = np.array(IR_train_Q12015,dtype=float)
IR_train_Q12015.resize((324050, 1))

x_train_Q12015 = np.concatenate((CS_train_Q12015,LTV_train_Q12015),axis=1)
x_train_Q12015 = np.concatenate((x_train_Q12015,DTI_train_Q12015),axis=1)
x_train_Q12015 = np.concatenate((x_train_Q12015,UPB_train_Q12015),axis=1)
x_train_Q12015 = np.concatenate((x_train_Q12015,IR_train_Q12015),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q12015 = np.array(DEFAULT_train_Q12015,dtype=int)
y_train_Q12015.resize((324050,))



# --------------------------------------------------
# Obtain the trainging data (Q2)
# --------------------------------------------------
dataQ22015 = xlrd.open_workbook("../../../Test Datasets/2015/data_Q22015.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ22015 = dataQ22015.sheet_by_name('Q22015')

# 获取工作表的名称，行数，列数
nameQ22015 = tableQ22015.name
rowNumQ22015 = tableQ22015.nrows
colNumQ22015 = tableQ22015.ncols

# 获取每个column的信息
CS_train_Q22015 = [float(tableQ22015.cell_value(i, 1)) for i in range(1, tableQ22015.nrows)]
LTV_train_Q22015 = [float(tableQ22015.cell_value(i, 2)) for i in range(1, tableQ22015.nrows)]
DTI_train_Q22015 = [float(tableQ22015.cell_value(i, 3)) for i in range(1, tableQ22015.nrows)]
UPB_train_Q22015 = [float(tableQ22015.cell_value(i, 4)) for i in range(1, tableQ22015.nrows)]
DEFAULT_train_Q22015 = [float(tableQ22015.cell_value(i, 5)) for i in range(1, tableQ22015.nrows)]
IR_train_Q22015 = [float(tableQ22015.cell_value(i, 6)) for i in range(1, tableQ22015.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q22015 = np.array(CS_train_Q22015,dtype=float)
CS_train_Q22015.resize((369332, 1))

LTV_train_Q22015 = np.array(LTV_train_Q22015,dtype=float)
LTV_train_Q22015.resize((369332, 1))

DTI_train_Q22015 = np.array(DTI_train_Q22015,dtype=float)
DTI_train_Q22015.resize((369332, 1))

UPB_train_Q22015 = np.array(UPB_train_Q22015,dtype=float)
UPB_train_Q22015.resize((369332, 1))

IR_train_Q22015 = np.array(IR_train_Q22015,dtype=float)
IR_train_Q22015.resize((369332, 1))

x_train_Q22015 = np.concatenate((CS_train_Q22015,LTV_train_Q22015),axis=1)
x_train_Q22015 = np.concatenate((x_train_Q22015,DTI_train_Q22015),axis=1)
x_train_Q22015 = np.concatenate((x_train_Q22015,UPB_train_Q22015),axis=1)
x_train_Q22015 = np.concatenate((x_train_Q22015,IR_train_Q22015),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q22015 = np.array(DEFAULT_train_Q22015,dtype=int)
y_train_Q22015.resize((369332,))

# --------------------------------------------------
# Obtain the trainging data (Q3)
# --------------------------------------------------
dataQ32015 = xlrd.open_workbook("../../../Test Datasets/2015/data_Q32015.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ32015 = dataQ32015.sheet_by_name('Q32015')

# 获取工作表的名称，行数，列数
nameQ32015 = tableQ32015.name
rowNumQ32015 = tableQ32015.nrows
colNumQ32015 = tableQ32015.ncols

# 获取每个column的信息
CS_train_Q32015 = [float(tableQ32015.cell_value(i, 1)) for i in range(1, tableQ32015.nrows)]
LTV_train_Q32015 = [float(tableQ32015.cell_value(i, 2)) for i in range(1, tableQ32015.nrows)]
DTI_train_Q32015 = [float(tableQ32015.cell_value(i, 3)) for i in range(1, tableQ32015.nrows)]
UPB_train_Q32015 = [float(tableQ32015.cell_value(i, 4)) for i in range(1, tableQ32015.nrows)]
DEFAULT_train_Q32015 = [float(tableQ32015.cell_value(i, 5)) for i in range(1, tableQ32015.nrows)]
IR_train_Q32015 = [float(tableQ32015.cell_value(i, 6)) for i in range(1, tableQ32015.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q32015 = np.array(CS_train_Q32015,dtype=float)
CS_train_Q32015.resize((332842, 1))

LTV_train_Q32015 = np.array(LTV_train_Q32015,dtype=float)
LTV_train_Q32015.resize((332842, 1))

DTI_train_Q32015 = np.array(DTI_train_Q32015,dtype=float)
DTI_train_Q32015.resize((332842, 1))

UPB_train_Q32015 = np.array(UPB_train_Q32015,dtype=float)
UPB_train_Q32015.resize((332842, 1))

IR_train_Q32015 = np.array(IR_train_Q32015,dtype=float)
IR_train_Q32015.resize((332842, 1))

x_train_Q32015 = np.concatenate((CS_train_Q32015,LTV_train_Q32015),axis=1)
x_train_Q32015 = np.concatenate((x_train_Q32015,DTI_train_Q32015),axis=1)
x_train_Q32015 = np.concatenate((x_train_Q32015,UPB_train_Q32015),axis=1)
x_train_Q32015 = np.concatenate((x_train_Q32015,IR_train_Q32015),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q32015 = np.array(DEFAULT_train_Q32015,dtype=int)
y_train_Q32015.resize((332842,))

# --------------------------------------------------
# Obtain the trainging data (Q4)
# --------------------------------------------------
dataQ42015 = xlrd.open_workbook("../../../Test Datasets/2015/data_Q42015.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ42015 = dataQ42015.sheet_by_name('Q42015')

# 获取工作表的名称，行数，列数
nameQ42015 = tableQ42015.name
rowNumQ42015 = tableQ42015.nrows
colNumQ42015 = tableQ42015.ncols

# 获取每个column的信息
CS_train_Q42015 = [float(tableQ42015.cell_value(i, 1)) for i in range(1, tableQ42015.nrows)]
LTV_train_Q42015 = [float(tableQ42015.cell_value(i, 2)) for i in range(1, tableQ42015.nrows)]
DTI_train_Q42015 = [float(tableQ42015.cell_value(i, 3)) for i in range(1, tableQ42015.nrows)]
UPB_train_Q42015 = [float(tableQ42015.cell_value(i, 4)) for i in range(1, tableQ42015.nrows)]
DEFAULT_train_Q42015 = [float(tableQ42015.cell_value(i, 5)) for i in range(1, tableQ42015.nrows)]
IR_train_Q42015 = [float(tableQ42015.cell_value(i, 6)) for i in range(1, tableQ42015.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q42015 = np.array(CS_train_Q42015,dtype=float)
CS_train_Q42015.resize((304825, 1))

LTV_train_Q42015 = np.array(LTV_train_Q42015,dtype=float)
LTV_train_Q42015.resize((304825, 1))

DTI_train_Q42015 = np.array(DTI_train_Q42015,dtype=float)
DTI_train_Q42015.resize((304825, 1))

UPB_train_Q42015 = np.array(UPB_train_Q42015,dtype=float)
UPB_train_Q42015.resize((304825, 1))

IR_train_Q42015 = np.array(IR_train_Q42015,dtype=float)
IR_train_Q42015.resize((304825, 1))

x_train_Q42015 = np.concatenate((CS_train_Q42015,LTV_train_Q42015),axis=1)
x_train_Q42015 = np.concatenate((x_train_Q42015,DTI_train_Q42015),axis=1)
x_train_Q42015 = np.concatenate((x_train_Q42015,UPB_train_Q42015),axis=1)
x_train_Q42015 = np.concatenate((x_train_Q42015,IR_train_Q42015),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q42015 = np.array(DEFAULT_train_Q42015,dtype=int)
y_train_Q42015.resize((304825,))

# Combine the 2015 training dataset
x_train_2015 = np.concatenate((x_train_Q12015,x_train_Q22015,x_train_Q32015,x_train_Q42015),axis=0)
y_train_2015 = np.concatenate((y_train_Q12015,y_train_Q22015,y_train_Q32015,y_train_Q42015),axis=0)

####################################################################
####################### Obtain 2016 Train Data #####################
####################################################################
# --------------------------------------------------
# Obtain the trainging data (Q1)
# --------------------------------------------------

dataQ12016 = xlrd.open_workbook("../../../Test Datasets/2016/data_Q12016.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ12016 = dataQ12016.sheet_by_name('Q12016')

# 获取工作表的名称，行数，列数
nameQ12016 = tableQ12016.name
rowNumQ12016 = tableQ12016.nrows
colNumQ12016 = tableQ12016.ncols

# 获取每个column的信息
CS_train_Q12016 = [float(tableQ12016.cell_value(i, 1)) for i in range(1, tableQ12016.nrows)]
LTV_train_Q12016 = [float(tableQ12016.cell_value(i, 2)) for i in range(1, tableQ12016.nrows)]
DTI_train_Q12016 = [float(tableQ12016.cell_value(i, 3)) for i in range(1, tableQ12016.nrows)]
UPB_train_Q12016 = [float(tableQ12016.cell_value(i, 4)) for i in range(1, tableQ12016.nrows)]
DEFAULT_train_Q12016 = [float(tableQ12016.cell_value(i, 5)) for i in range(1, tableQ12016.nrows)]
IR_train_Q12016 = [float(tableQ12016.cell_value(i, 6)) for i in range(1, tableQ12016.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q12016 = np.array(CS_train_Q12016,dtype=float)
CS_train_Q12016.resize((280188, 1))

LTV_train_Q12016 = np.array(LTV_train_Q12016,dtype=float)
LTV_train_Q12016.resize((280188, 1))

DTI_train_Q12016 = np.array(DTI_train_Q12016,dtype=float)
DTI_train_Q12016.resize((280188, 1))

UPB_train_Q12016 = np.array(UPB_train_Q12016,dtype=float)
UPB_train_Q12016.resize((280188, 1))

IR_train_Q12016 = np.array(IR_train_Q12016,dtype=float)
IR_train_Q12016.resize((280188, 1))

x_train_Q12016 = np.concatenate((CS_train_Q12016,LTV_train_Q12016),axis=1)
x_train_Q12016 = np.concatenate((x_train_Q12016,DTI_train_Q12016),axis=1)
x_train_Q12016 = np.concatenate((x_train_Q12016,UPB_train_Q12016),axis=1)
x_train_Q12016 = np.concatenate((x_train_Q12016,IR_train_Q12016),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q12016 = np.array(DEFAULT_train_Q12016,dtype=int)
y_train_Q12016.resize((280188,))



# --------------------------------------------------
# Obtain the trainging data (Q2)
# --------------------------------------------------
dataQ22016 = xlrd.open_workbook("../../../Test Datasets/2016/data_Q22016.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ22016 = dataQ22016.sheet_by_name('Q22016')

# 获取工作表的名称，行数，列数
nameQ22016 = tableQ22016.name
rowNumQ22016 = tableQ22016.nrows
colNumQ22016 = tableQ22016.ncols

# 获取每个column的信息
CS_train_Q22016 = [float(tableQ22016.cell_value(i, 1)) for i in range(1, tableQ22016.nrows)]
LTV_train_Q22016 = [float(tableQ22016.cell_value(i, 2)) for i in range(1, tableQ22016.nrows)]
DTI_train_Q22016 = [float(tableQ22016.cell_value(i, 3)) for i in range(1, tableQ22016.nrows)]
UPB_train_Q22016 = [float(tableQ22016.cell_value(i, 4)) for i in range(1, tableQ22016.nrows)]
DEFAULT_train_Q22016 = [float(tableQ22016.cell_value(i, 5)) for i in range(1, tableQ22016.nrows)]
IR_train_Q22016 = [float(tableQ22016.cell_value(i, 6)) for i in range(1, tableQ22016.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q22016 = np.array(CS_train_Q22016,dtype=float)
CS_train_Q22016.resize((400346, 1))

LTV_train_Q22016 = np.array(LTV_train_Q22016,dtype=float)
LTV_train_Q22016.resize((400346, 1))

DTI_train_Q22016 = np.array(DTI_train_Q22016,dtype=float)
DTI_train_Q22016.resize((400346, 1))

UPB_train_Q22016 = np.array(UPB_train_Q22016,dtype=float)
UPB_train_Q22016.resize((400346, 1))

IR_train_Q22016 = np.array(IR_train_Q22016,dtype=float)
IR_train_Q22016.resize((400346, 1))

x_train_Q22016 = np.concatenate((CS_train_Q22016,LTV_train_Q22016),axis=1)
x_train_Q22016 = np.concatenate((x_train_Q22016,DTI_train_Q22016),axis=1)
x_train_Q22016 = np.concatenate((x_train_Q22016,UPB_train_Q22016),axis=1)
x_train_Q22016 = np.concatenate((x_train_Q22016,IR_train_Q22016),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q22016 = np.array(DEFAULT_train_Q22016,dtype=int)
y_train_Q22016.resize((400346,))

# --------------------------------------------------
# Obtain the trainging data (Q3)
# --------------------------------------------------
dataQ32016 = xlrd.open_workbook("../../../Test Datasets/2016/data_Q32016.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ32016 = dataQ32016.sheet_by_name('Q32016')

# 获取工作表的名称，行数，列数
nameQ32016 = tableQ32016.name
rowNumQ32016 = tableQ32016.nrows
colNumQ32016 = tableQ32016.ncols

# 获取每个column的信息
CS_train_Q32016 = [float(tableQ32016.cell_value(i, 1)) for i in range(1, tableQ32016.nrows)]
LTV_train_Q32016 = [float(tableQ32016.cell_value(i, 2)) for i in range(1, tableQ32016.nrows)]
DTI_train_Q32016 = [float(tableQ32016.cell_value(i, 3)) for i in range(1, tableQ32016.nrows)]
UPB_train_Q32016 = [float(tableQ32016.cell_value(i, 4)) for i in range(1, tableQ32016.nrows)]
DEFAULT_train_Q32016 = [float(tableQ32016.cell_value(i, 5)) for i in range(1, tableQ32016.nrows)]
IR_train_Q32016 = [float(tableQ32016.cell_value(i, 6)) for i in range(1, tableQ32016.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q32016 = np.array(CS_train_Q32016,dtype=float)
CS_train_Q32016.resize((446342, 1))

LTV_train_Q32016 = np.array(LTV_train_Q32016,dtype=float)
LTV_train_Q32016.resize((446342, 1))

DTI_train_Q32016 = np.array(DTI_train_Q32016,dtype=float)
DTI_train_Q32016.resize((446342, 1))

UPB_train_Q32016 = np.array(UPB_train_Q32016,dtype=float)
UPB_train_Q32016.resize((446342, 1))

IR_train_Q32016 = np.array(IR_train_Q32016,dtype=float)
IR_train_Q32016.resize((446342, 1))

x_train_Q32016 = np.concatenate((CS_train_Q32016,LTV_train_Q32016),axis=1)
x_train_Q32016 = np.concatenate((x_train_Q32016,DTI_train_Q32016),axis=1)
x_train_Q32016 = np.concatenate((x_train_Q32016,UPB_train_Q32016),axis=1)
x_train_Q32016 = np.concatenate((x_train_Q32016,IR_train_Q32016),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q32016 = np.array(DEFAULT_train_Q32016,dtype=int)
y_train_Q32016.resize((446342,))

# --------------------------------------------------
# Obtain the trainging data (Q4)
# --------------------------------------------------
dataQ42016 = xlrd.open_workbook("../../../Test Datasets/2016/data_Q42016.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ42016 = dataQ42016.sheet_by_name('Q42016')

# 获取工作表的名称，行数，列数
nameQ42016 = tableQ42016.name
rowNumQ42016 = tableQ42016.nrows
colNumQ42016 = tableQ42016.ncols

# 获取每个column的信息
CS_train_Q42016 = [float(tableQ42016.cell_value(i, 1)) for i in range(1, tableQ42016.nrows)]
LTV_train_Q42016 = [float(tableQ42016.cell_value(i, 2)) for i in range(1, tableQ42016.nrows)]
DTI_train_Q42016 = [float(tableQ42016.cell_value(i, 3)) for i in range(1, tableQ42016.nrows)]
UPB_train_Q42016 = [float(tableQ42016.cell_value(i, 4)) for i in range(1, tableQ42016.nrows)]
DEFAULT_train_Q42016 = [float(tableQ42016.cell_value(i, 5)) for i in range(1, tableQ42016.nrows)]
IR_train_Q42016 = [float(tableQ42016.cell_value(i, 6)) for i in range(1, tableQ42016.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q42016 = np.array(CS_train_Q42016,dtype=float)
CS_train_Q42016.resize((436630, 1))

LTV_train_Q42016 = np.array(LTV_train_Q42016,dtype=float)
LTV_train_Q42016.resize((436630, 1))

DTI_train_Q42016 = np.array(DTI_train_Q42016,dtype=float)
DTI_train_Q42016.resize((436630, 1))

UPB_train_Q42016 = np.array(UPB_train_Q42016,dtype=float)
UPB_train_Q42016.resize((436630, 1))

IR_train_Q42016 = np.array(IR_train_Q42016,dtype=float)
IR_train_Q42016.resize((436630, 1))

x_train_Q42016 = np.concatenate((CS_train_Q42016,LTV_train_Q42016),axis=1)
x_train_Q42016 = np.concatenate((x_train_Q42016,DTI_train_Q42016),axis=1)
x_train_Q42016 = np.concatenate((x_train_Q42016,UPB_train_Q42016),axis=1)
x_train_Q42016 = np.concatenate((x_train_Q42016,IR_train_Q42016),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q42016 = np.array(DEFAULT_train_Q42016,dtype=int)
y_train_Q42016.resize((436630,))

# Combine the 2016 training dataset
x_train_2016 = np.concatenate((x_train_Q12016,x_train_Q22016,x_train_Q32016,x_train_Q42016),axis=0)
y_train_2016 = np.concatenate((y_train_Q12016,y_train_Q22016,y_train_Q32016,y_train_Q42016),axis=0)



####################################################################
####################### Obtain 2017 Train Data #####################
####################################################################
# --------------------------------------------------
# Obtain the trainging data (Q1)
# --------------------------------------------------

dataQ12017 = xlrd.open_workbook("../../../Test Datasets/2017/data_Q12017.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ12017 = dataQ12017.sheet_by_name('Q12017')

# 获取工作表的名称，行数，列数
nameQ12017 = tableQ12017.name
rowNumQ12017 = tableQ12017.nrows
colNumQ12017 = tableQ12017.ncols

# 获取每个column的信息
CS_train_Q12017 = [float(tableQ12017.cell_value(i, 1)) for i in range(1, tableQ12017.nrows)]
LTV_train_Q12017 = [float(tableQ12017.cell_value(i, 2)) for i in range(1, tableQ12017.nrows)]
DTI_train_Q12017 = [float(tableQ12017.cell_value(i, 3)) for i in range(1, tableQ12017.nrows)]
UPB_train_Q12017 = [float(tableQ12017.cell_value(i, 4)) for i in range(1, tableQ12017.nrows)]
DEFAULT_train_Q12017 = [float(tableQ12017.cell_value(i, 5)) for i in range(1, tableQ12017.nrows)]
IR_train_Q12017 = [float(tableQ12017.cell_value(i, 6)) for i in range(1, tableQ12017.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q12017 = np.array(CS_train_Q12017,dtype=float)
CS_train_Q12017.resize((256883, 1))

LTV_train_Q12017 = np.array(LTV_train_Q12017,dtype=float)
LTV_train_Q12017.resize((256883, 1))

DTI_train_Q12017 = np.array(DTI_train_Q12017,dtype=float)
DTI_train_Q12017.resize((256883, 1))

UPB_train_Q12017 = np.array(UPB_train_Q12017,dtype=float)
UPB_train_Q12017.resize((256883, 1))

IR_train_Q12017 = np.array(IR_train_Q12017,dtype=float)
IR_train_Q12017.resize((256883, 1))

x_train_Q12017 = np.concatenate((CS_train_Q12017,LTV_train_Q12017),axis=1)
x_train_Q12017 = np.concatenate((x_train_Q12017,DTI_train_Q12017),axis=1)
x_train_Q12017 = np.concatenate((x_train_Q12017,UPB_train_Q12017),axis=1)
x_train_Q12017 = np.concatenate((x_train_Q12017,IR_train_Q12017),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q12017 = np.array(DEFAULT_train_Q12017,dtype=int)
y_train_Q12017.resize((256883,))

# --------------------------------------------------
# Obtain the trainging data (Q2)
# --------------------------------------------------
dataQ22017 = xlrd.open_workbook("../../../Test Datasets/2017/data_Q22017.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ22017 = dataQ22017.sheet_by_name('Q22017')

# 获取工作表的名称，行数，列数
nameQ22017 = tableQ22017.name
rowNumQ22017 = tableQ22017.nrows
colNumQ22017 = tableQ22017.ncols

# 获取每个column的信息
CS_train_Q22017 = [float(tableQ22017.cell_value(i, 1)) for i in range(1, tableQ22017.nrows)]
LTV_train_Q22017 = [float(tableQ22017.cell_value(i, 2)) for i in range(1, tableQ22017.nrows)]
DTI_train_Q22017 = [float(tableQ22017.cell_value(i, 3)) for i in range(1, tableQ22017.nrows)]
UPB_train_Q22017 = [float(tableQ22017.cell_value(i, 4)) for i in range(1, tableQ22017.nrows)]
DEFAULT_train_Q22017 = [float(tableQ22017.cell_value(i, 5)) for i in range(1, tableQ22017.nrows)]
IR_train_Q22017 = [float(tableQ22017.cell_value(i, 6)) for i in range(1, tableQ22017.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q22017 = np.array(CS_train_Q22017,dtype=float)
CS_train_Q22017.resize((296582, 1))

LTV_train_Q22017 = np.array(LTV_train_Q22017,dtype=float)
LTV_train_Q22017.resize((296582, 1))

DTI_train_Q22017 = np.array(DTI_train_Q22017,dtype=float)
DTI_train_Q22017.resize((296582, 1))

UPB_train_Q22017 = np.array(UPB_train_Q22017,dtype=float)
UPB_train_Q22017.resize((296582, 1))

IR_train_Q22017 = np.array(IR_train_Q22017,dtype=float)
IR_train_Q22017.resize((296582, 1))

x_train_Q22017 = np.concatenate((CS_train_Q22017,LTV_train_Q22017),axis=1)
x_train_Q22017 = np.concatenate((x_train_Q22017,DTI_train_Q22017),axis=1)
x_train_Q22017 = np.concatenate((x_train_Q22017,UPB_train_Q22017),axis=1)
x_train_Q22017 = np.concatenate((x_train_Q22017,IR_train_Q22017),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q22017 = np.array(DEFAULT_train_Q22017,dtype=int)
y_train_Q22017.resize((296582,))

# --------------------------------------------------
# Obtain the trainging data (Q3)
# --------------------------------------------------
dataQ32017 = xlrd.open_workbook("../../../Test Datasets/2017/data_Q32017.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ32017 = dataQ32017.sheet_by_name('Q32017')

# 获取工作表的名称，行数，列数
nameQ32017 = tableQ32017.name
rowNumQ32017 = tableQ32017.nrows
colNumQ32017 = tableQ32017.ncols

# 获取每个column的信息
CS_train_Q32017 = [float(tableQ32017.cell_value(i, 1)) for i in range(1, tableQ32017.nrows)]
LTV_train_Q32017 = [float(tableQ32017.cell_value(i, 2)) for i in range(1, tableQ32017.nrows)]
DTI_train_Q32017 = [float(tableQ32017.cell_value(i, 3)) for i in range(1, tableQ32017.nrows)]
UPB_train_Q32017 = [float(tableQ32017.cell_value(i, 4)) for i in range(1, tableQ32017.nrows)]
DEFAULT_train_Q32017 = [float(tableQ32017.cell_value(i, 5)) for i in range(1, tableQ32017.nrows)]
IR_train_Q32017 = [float(tableQ32017.cell_value(i, 6)) for i in range(1, tableQ32017.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q32017 = np.array(CS_train_Q32017,dtype=float)
CS_train_Q32017.resize((351733, 1))

LTV_train_Q32017 = np.array(LTV_train_Q32017,dtype=float)
LTV_train_Q32017.resize((351733, 1))

DTI_train_Q32017 = np.array(DTI_train_Q32017,dtype=float)
DTI_train_Q32017.resize((351733, 1))

UPB_train_Q32017 = np.array(UPB_train_Q32017,dtype=float)
UPB_train_Q32017.resize((351733, 1))

IR_train_Q32017 = np.array(IR_train_Q32017,dtype=float)
IR_train_Q32017.resize((351733, 1))

x_train_Q32017 = np.concatenate((CS_train_Q32017,LTV_train_Q32017),axis=1)
x_train_Q32017 = np.concatenate((x_train_Q32017,DTI_train_Q32017),axis=1)
x_train_Q32017 = np.concatenate((x_train_Q32017,UPB_train_Q32017),axis=1)
x_train_Q32017 = np.concatenate((x_train_Q32017,IR_train_Q32017),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q32017 = np.array(DEFAULT_train_Q32017,dtype=int)
y_train_Q32017.resize((351733,))

# --------------------------------------------------
# Obtain the trainging data (Q4)
# --------------------------------------------------
dataQ42017 = xlrd.open_workbook("../../../Test Datasets/2017/data_Q42017.xlsx")

# 根据工作表名称获取里面的行列内容
tableQ42017 = dataQ42017.sheet_by_name('Q42017')

# 获取工作表的名称，行数，列数
nameQ42017 = tableQ42017.name
rowNumQ42017 = tableQ42017.nrows
colNumQ42017 = tableQ42017.ncols

# 获取每个column的信息
CS_train_Q42017 = [float(tableQ42017.cell_value(i, 1)) for i in range(1, tableQ42017.nrows)]
LTV_train_Q42017 = [float(tableQ42017.cell_value(i, 2)) for i in range(1, tableQ42017.nrows)]
DTI_train_Q42017 = [float(tableQ42017.cell_value(i, 3)) for i in range(1, tableQ42017.nrows)]
UPB_train_Q42017 = [float(tableQ42017.cell_value(i, 4)) for i in range(1, tableQ42017.nrows)]
DEFAULT_train_Q42017 = [float(tableQ42017.cell_value(i, 5)) for i in range(1, tableQ42017.nrows)]
IR_train_Q42017 = [float(tableQ42017.cell_value(i, 6)) for i in range(1, tableQ42017.nrows)]

# Reshape and turn the list into arr (X)
import numpy as np
CS_train_Q42017 = np.array(CS_train_Q42017,dtype=float)
CS_train_Q42017.resize((322033, 1))

LTV_train_Q42017 = np.array(LTV_train_Q42017,dtype=float)
LTV_train_Q42017.resize((322033, 1))

DTI_train_Q42017 = np.array(DTI_train_Q42017,dtype=float)
DTI_train_Q42017.resize((322033, 1))

UPB_train_Q42017 = np.array(UPB_train_Q42017,dtype=float)
UPB_train_Q42017.resize((322033, 1))

IR_train_Q42017 = np.array(IR_train_Q42017,dtype=float)
IR_train_Q42017.resize((322033, 1))

x_train_Q42017 = np.concatenate((CS_train_Q42017,LTV_train_Q42017),axis=1)
x_train_Q42017 = np.concatenate((x_train_Q42017,DTI_train_Q42017),axis=1)
x_train_Q42017 = np.concatenate((x_train_Q42017,UPB_train_Q42017),axis=1)
x_train_Q42017 = np.concatenate((x_train_Q42017,IR_train_Q42017),axis=1)

# Reshape and turn the list into arr(y)
y_train_Q42017 = np.array(DEFAULT_train_Q42017,dtype=int)
y_train_Q42017.resize((322033,))

# Combine the 2017 training dataset
x_train_2017 = np.concatenate((x_train_Q12017,x_train_Q22017,x_train_Q32017,x_train_Q42017),axis=0)
y_train_2017 = np.concatenate((y_train_Q12017,y_train_Q22017,y_train_Q32017,y_train_Q42017),axis=0)


# Combine the three train dataset as one
x_train_combined = np.concatenate((x_train_2015,x_train_2016,x_train_2017),axis=0)
y_train_combined = np.concatenate((y_train_2015,y_train_2016,y_train_2017),axis=0)

################################################################################
# Build Model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
#建模
# The training accurancy
model_balanced = GaussianNB()
model_balanced.fit(x_train_combined,y_train_combined)
# save the model to disk
filename = 'Naive Bayes Trained Model.sav'
pickle.dump(model_balanced, open(filename, 'wb'))
 