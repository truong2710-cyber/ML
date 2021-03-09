import xlrd
import numpy as np

def extractFeature(student):
    v=[]
    s=student.split(";")
    if s[0]=="GP": v.append(0) 
    else: v.append(1)
    if s[1]=='"F"': v.append(0) 
    else: v.append(1)
    v.append(int(s[2]))
    if s[3]=='"U"': v.append(0) 
    else: v.append(1)
    if s[4]=='"LT3"': v.append(0) 
    else: v.append(1)
    if s[5]=='"T"': v.append(0) 
    else: v.append(1)
    v.append(int(s[6]))
    v.append(int(s[7]))
    if s[8]=='"teacher"': v.append(0)
    elif s[8]=='"health"': v.append(1)
    elif s[8]=='"services"': v.append(2)
    elif s[8]=='"at_home"': v.append(3)
    else: v.append(4)
    if s[9]=='"teacher"': v.append(0)
    elif s[9]=='"health"': v.append(1)
    elif s[9]=='"services"': v.append(2)
    elif s[9]=='"at_home"': v.append(3)
    else: v.append(4)
    if s[10]=='"home"': v.append(0)
    elif s[10]=='"reputation"': v.append(1)
    elif s[10]=='"course"': v.append(2)
    else: v.append(3)
    if s[11]=='"mother"': v.append(0)
    elif s[11]=='"father"': v.append(1)
    else: v.append(2)
    v.append(int(s[12]))
    v.append(int(s[13]))
    v.append(int(s[14]))
    if s[15]=='"no"': v.append(0) 
    else: v.append(1)
    if s[16]=='"no"': v.append(0) 
    else: v.append(1)
    if s[17]=='"no"': v.append(0) 
    else: v.append(1)
    if s[18]=='"no"': v.append(0) 
    else: v.append(1)
    if s[19]=='"no"': v.append(0) 
    else: v.append(1)
    if s[20]=='"no"': v.append(0) 
    else: v.append(1)
    if s[21]=='"no"': v.append(0) 
    else: v.append(1)
    if s[22]=='"no"': v.append(0) 
    else: v.append(1)
    v.append(int(s[23]))
    v.append(int(s[24]))
    v.append(int(s[25]))
    v.append(int(s[26]))
    v.append(int(s[27]))
    v.append(int(s[28]))
    v.append(int(s[29]))
    v.append(int(s[30].strip('"')))
    v.append(int(s[31].strip('"')))
    return v

def output(student):
    s=student.split(";")
    return int(s[32])

#train 
students=[]
outputgrade=[]
wb=xlrd.open_workbook("D:\\Python\\ML\\student\\student-mat.xlsx")
sheet=wb.sheet_by_index(0)
N=350 #training set
for i in range(1,N+1):
    students.append(extractFeature(sheet.cell_value(i, 0)))
for i in range(1,396):    
    outputgrade.append(output(sheet.cell_value(i, 0)))
X=np.array(students)
y=np.array(outputgrade[:N]).T
Xbar=np.concatenate((np.ones((N,1)),X),axis=1)
w=np.linalg.pinv(Xbar.T.dot(Xbar)).dot(Xbar.T.dot(y))
print(w.T)

#test
Xtest=[]
for i in range(N+1,396):
    Xtest.append(extractFeature(sheet.cell_value(i, 0)))
Xtest=np.array(Xtest)
Xtestbar=np.concatenate((np.ones((395-N,1)),Xtest),axis=1)
ytest=np.rint(Xtestbar.dot(w))
original_output=np.array(outputgrade[N:396]).T
print(original_output.T)
print(ytest.T)
score = 1-((original_output - ytest) ** 2).sum()/((original_output - original_output.mean()) ** 2).sum()
print("Accuracy core: %.2f%%"%(100*score))
