import numpy as np

t = open('F:\python\KNN\label_SBP_test.txt','w')
#t = open('label_test.txt','w')
num_1=24
num_2=24
label1=np.ones((num_1,1))
label2=np.zeros((num_2,1))

for i in range(num_1):
    zifu='000000'
    zifuchuan=zifu+str(i+1)+' '+str(int(label1[i]))
    t.write(zifuchuan +'\n')
    zifu=[]
    da1=[]
    
for j in range(num_2):
    zifu='000000'
    zifuchuan=zifu+str(j+num_1+1)+' '+str(int(label2[j]))
    t.write(zifuchuan +'\n')
    zifu=[]
    da1=[]
t.close()
