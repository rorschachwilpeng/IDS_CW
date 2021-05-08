import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np
import xlrd
from sklearn.linear_model import LogisticRegression

######################################################   
# Q12018
###################################################### 
data1 = xlrd.open_workbook("../../Test DataSets/2018/data_Q12018.xlsx")

# 根据工作表名称获取里面的行列内容
table1 = data1.sheet_by_name('Q12018')

# 获取工作表的名称，行数，列数
name1 = table1.name
rowNum1 = table1.nrows
colNum1 = table1.ncols

# 获取每个column的信息
CS = [float(table1.cell_value(i, 1)) for i in range(1, table1.nrows)]
LTV = [float(table1.cell_value(i, 2)) for i in range(1, table1.nrows)]
DTI = [float(table1.cell_value(i, 3)) for i in range(1, table1.nrows)]
UPB = [float(table1.cell_value(i, 4)) for i in range(1, table1.nrows)]
DEFAULT = [float(table1.cell_value(i, 5)) for i in range(1, table1.nrows)]
IR = [float(table1.cell_value(i, 6)) for i in range(1, table1.nrows)]

import pandas as pd
# 将list转换成可以直接用相关系数函数调用的dataframe
df_grade1 = pd.DataFrame(CS, columns=['CS'])
df_grade2 = pd.DataFrame(LTV, columns=['LTV'])
df_grade3 = pd.DataFrame(DTI, columns=['DTI'])
df_grade4 = pd.DataFrame(UPB, columns=['UPB'])
df_grade5 = pd.DataFrame(DEFAULT, columns=['DEFAULT'])
df_grade6 = pd.DataFrame(IR, columns=['IR'])

#DataFrame
df_grade_1 = pd.concat([df_grade1,df_grade2,df_grade3,df_grade4,df_grade5,df_grade6],axis=1)
#df_grade

######################################################   
# Q22018
###################################################### 
data2 = xlrd.open_workbook("../../Test DataSets/2018/data_Q22018.xlsx")

# 根据工作表名称获取里面的行列内容
table2 = data2.sheet_by_name('Q22018')

# 获取工作表的名称，行数，列数
name2 = table2.name
rowNum2 = table2.nrows
colNum2 = table2.ncols

# 获取每个column的信息
CS = [float(table2.cell_value(i, 1)) for i in range(1, table2.nrows)]
LTV = [float(table2.cell_value(i, 2)) for i in range(1, table2.nrows)]
DTI = [float(table2.cell_value(i, 3)) for i in range(1, table2.nrows)]
UPB = [float(table2.cell_value(i, 4)) for i in range(1, table2.nrows)]
DEFAULT = [float(table2.cell_value(i, 5)) for i in range(1, table2.nrows)]
IR = [float(table2.cell_value(i, 6)) for i in range(1, table2.nrows)]

import pandas as pd
# 将list转换成可以直接用相关系数函数调用的dataframe
df_grade1 = pd.DataFrame(CS, columns=['CS'])
df_grade2 = pd.DataFrame(LTV, columns=['LTV'])
df_grade3 = pd.DataFrame(DTI, columns=['DTI'])
df_grade4 = pd.DataFrame(UPB, columns=['UPB'])
df_grade5 = pd.DataFrame(DEFAULT, columns=['DEFAULT'])
df_grade6 = pd.DataFrame(IR, columns=['IR'])

#DataFrame
df_grade_2 = pd.concat([df_grade1,df_grade2,df_grade3,df_grade4,df_grade5,df_grade6],axis=1)
######################################################   
# Q32018
###################################################### 
data3 = xlrd.open_workbook("../../Test DataSets/2018/data_Q32018.xlsx")

# 根据工作表名称获取里面的行列内容
table3 = data3.sheet_by_name('Q32018')

# 获取工作表的名称，行数，列数
name3 = table3.name
rowNum3 = table3.nrows
colNum3 = table3.ncols

# 获取每个column的信息
CS = [float(table3.cell_value(i, 1)) for i in range(1, table3.nrows)]
LTV = [float(table3.cell_value(i, 2)) for i in range(1, table3.nrows)]
DTI = [float(table3.cell_value(i, 3)) for i in range(1, table3.nrows)]
UPB = [float(table3.cell_value(i, 4)) for i in range(1, table3.nrows)]
DEFAULT = [float(table3.cell_value(i, 5)) for i in range(1, table3.nrows)]
IR = [float(table3.cell_value(i, 6)) for i in range(1, table3.nrows)]

import pandas as pd
# 将list转换成可以直接用相关系数函数调用的dataframe
df_grade1 = pd.DataFrame(CS, columns=['CS'])
df_grade2 = pd.DataFrame(LTV, columns=['LTV'])
df_grade3 = pd.DataFrame(DTI, columns=['DTI'])
df_grade4 = pd.DataFrame(UPB, columns=['UPB'])
df_grade5 = pd.DataFrame(DEFAULT, columns=['DEFAULT'])
df_grade6 = pd.DataFrame(IR, columns=['IR'])

#DataFrame
df_grade_3 = pd.concat([df_grade1,df_grade2,df_grade3,df_grade4,df_grade5,df_grade6],axis=1)

######################################################   
# Q42018
###################################################### 
data4 = xlrd.open_workbook("../../Test DataSets/2018/data_Q42018.xlsx")

# 根据工作表名称获取里面的行列内容
table4 = data4.sheet_by_name('Q42018')

# 获取工作表的名称，行数，列数
name4 = table4.name
rowNum4 = table4.nrows
colNum4 = table4.ncols

# 获取每个column的信息
CS = [float(table4.cell_value(i, 1)) for i in range(1, table4.nrows)]
LTV = [float(table4.cell_value(i, 2)) for i in range(1, table4.nrows)]
DTI = [float(table4.cell_value(i, 3)) for i in range(1, table4.nrows)]
UPB = [float(table4.cell_value(i, 4)) for i in range(1, table4.nrows)]
DEFAULT = [float(table4.cell_value(i, 5)) for i in range(1, table4.nrows)]
IR = [float(table4.cell_value(i, 6)) for i in range(1, table4.nrows)]

import pandas as pd
# 将list转换成可以直接用相关系数函数调用的dataframe
df_grade1 = pd.DataFrame(CS, columns=['CS'])
df_grade2 = pd.DataFrame(LTV, columns=['LTV'])
df_grade3 = pd.DataFrame(DTI, columns=['DTI'])
df_grade4 = pd.DataFrame(UPB, columns=['UPB'])
df_grade5 = pd.DataFrame(DEFAULT, columns=['DEFAULT'])
df_grade6 = pd.DataFrame(IR, columns=['IR'])

#DataFrame
df_grade_4 = pd.concat([df_grade1,df_grade2,df_grade3,df_grade4,df_grade5,df_grade6],axis=1)

######################################################  Combined all the data in 2018 into one dataframe
# df_grade_1d
# df_grade_2
# df_grade_3
# df_grade_4
# Combined all the data in 2018 into one dataframe
combined_df = df_grade_1.append([df_grade_2, df_grade_3,df_grade_4])
# combined_df

######################################################
# Pie Chart in Report 
######################################################
# Count the number of each "default"
num_zero =0
num_one = 0
for item in combined_df['DEFAULT']:
    if(item == 0):
        num_zero+=1
    else:
        num_one+=1
plt.figure(figsize=(6,9)) #调节图形大小
labels = [u'Default value is 0',u'Default value is 1'] #定义标签
sizes = [num_zero,num_one] #每块值
colors = ['red','yellowgreen'] #每块颜色定义
explode = (0,0) #将某一块分割出来，值越大分割出的间隙越大
patches,text1,text2 = plt.pie(sizes,
                    explode=explode,
                    labels=labels,
                    colors=colors,
                    labeldistance = 1.2,#图例距圆心半径倍距离
                    autopct = '%3.2f%%', #数值保留固定小数位
                    shadow = False, #无阴影设置
                    startangle =90, #逆时针起始角度设置
                    pctdistance = 0.6) #数值距圆心半径倍数距离
#patches饼图的返回值，texts1饼图外label的文本，texts2饼图内部文本
# x，y轴刻度设置一致，保证饼图为圆形
plt.axis('equal')
plt.legend()
# plt.show()
plt.savefig('./OUTPUT/Pie_Chart.png')

######################################################
# Bar Chart in report 
num_zero =0
num_one = 0
for item in combined_df['DEFAULT']:
    if(item == 0):
        num_zero+=1
    else:
        num_one+=1
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
name_list = ['Default value is 0','Default value is 1']
num_list = [num_zero,num_one]
plt.bar(range(len(num_list)), num_list,color='rgb',tick_label=name_list)
plt.savefig('./OUTPUT/Bar_Chart.png',dpi=100)

######################################################
# Covariance matrix in report
import pandas as pd
a = np.array(combined_df['CS'])
b = np.array(combined_df['LTV'])
c = np.array(combined_df['DTI'])
d = np.array(combined_df['UPB'])
e = np.array(combined_df['DEFAULT'])

cc_pd = pd.DataFrame(combined_df)
cc_corr = cc_pd.corr(method='spearman')   #相关系数矩阵

cc_corr.to_excel('./OUTPUT/Spearman_Corr_Martix.xls')

######################################################
# Scaled coariance matrix in report
(cc_corr['DEFAULT']*100).to_excel('./OUTPUT/Scale_Spearman_Corr_Martix.xls')

######################################################
# HeatMap in report
import seaborn as sns
import matplotlib.pyplot as plt
# regular_df = pd.DataFrame(X_normalized)
# regular_df.columns = ['CS','LTV','DTI','UPB','DEFAULT','IR']
figure,ax = plt.subplots(figsize=(12,12))
sns.heatmap(cc_corr,square=True, annot=True,ax=ax)
plt.savefig('./OUTPUT/Speraman_Regular_Heat_Map.png')
