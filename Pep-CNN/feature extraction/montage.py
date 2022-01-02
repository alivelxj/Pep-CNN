import numpy as np
import pandas as pd

#读取数据
dataset1 = pd.read_excel(r'C:\BE_AAP.xlsx', header = None)
dataset2 = pd.read_excel(r'C:\EBGW_AAP.xlsx', header = None)
dataset3 = pd.read_csv(r'C:\EGAAC_AAP.csv', header = None)
dataset4 = pd.read_csv(r'C:\BLOSUM62_AAP.csv', header = None)
dataset5 = pd.read_csv(r'C:\KNN_AAP.csv', header = None)
#dataset6=pd.read_csv('C:/Users/HP/Desktop/5hmc/BKF.csv',header=None)
#dataset7 =pd.read_csv('C:/Users/HP/Desktop/5hmc/X_DBPF.csv',header=None)
#dataset8 =pd.read_excel('C:/Users/HP/Desktop/5hmc/mis.xlsx',header=None)
#dataset9 =pd.read_csv('C:/Users/HP/Desktop/5hmc/misbz.csv',header=None)
#dataset10 =pd.read_excel('C:/Users/HP/Desktop/5hmc/GC.xlsx',header=None)
#dataset11 =pd.read_excel('C:/Users/HP/Desktop/5hmc/MAC.xlsx',header=None)
#dataset12 = pd.read_csv('C:/Users/HP/Desktop/5hmc/X_PCP.csv',header=None)
#dataset13 = pd.read_excel('C:/Users/HP/Desktop/5hmc/sub.xlsx',header=None)
#dataset14 = pd.read_excel('C:/Users/HP/Desktop/5hmc/pseSSC.xlsx',header=None)
#dataset6=pd.read_csv('C:/Users/HP/Desktop/5hmc/BKF.csv',header=None)

#dataset12 =pd.read_excel('C:/Users/HP/Desktop/5hmc/融合/4.xlsx',header=None)
#特征数据拼接
dataset = np.column_stack((dataset1,dataset2,dataset3,dataset4,dataset5))

#将数据保存至csv文件
pd.DataFrame(dataset).to_csv(r'C:\feature_AAP.csv',header = None, index = False)

#将数据保存至excel
# from openpyxl import *
# wb = Workbook()
# ws = wb.active
# for row in range(1,dataset.shape[0]+1):
#     for col in range(1,dataset.shape[1]+1):
#         ws.cell(row=row,column=col,value=dataset[row-1][col-1])
# wb.save("11111.xlsx")