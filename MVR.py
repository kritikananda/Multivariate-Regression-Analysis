#Kritika Nanda
#101903092

#code for multivariate regression.
import csv
#csv - comma separated values used to import spreadsheets and databases
import math
#mathematical functions
import numpy as np
#numpy to work with arrays
import matplotlib.pyplot as plt
#plotting graphs

#https://vincentarelbundock.github.io/Rdatasets/csv/AER/CASchools.csv
#Obtaining data from the dataset
fileobj=open('CASchools.csv')
filereader=csv.reader(fileobj,delimiter=',')

table=list(filereader)  #data arranged in a list
table=np.array(table)   #data is stored in the form of array

#A data frame containing 420 observations on 14 variables.

#MultiVariateRegression components
n=table.shape[0]-1      #total number of observations
q=2     #Number of y
p=8     #Number of x

print("The dataset has 2 dependent variables and 12 independent variables")
#dependent are - read ;Average reading score and math ;Average math score.
#independent are - district,school,county,grades,students,teachers,calworks,lunch,computer,expenditure,income,english

#Matrix form of MVLR
#Y=XB
Y=[[0.000 for i in range(q)]for j in range(n)]
Y=np.array(Y)
#Y n*q
X=[[1.000 for i in range(p+1)]for j in range(n)]
X=np.array(X)
#X n*(p+1)
beta=[[0.000 for i in range(q)]for j in range(p+1)]
beta=np.array(beta)
#B (p+1)*q

#Y
for i in range(1,n+1,1):
    for j in range(13,15,1):    #13 and 14th column of the dataset
        Y[i-1,j-13]=float(table[i,j])
#X        
for i in range(1,n+1,1):
    for j in range(4,13,1):     #4th to 12th column of the dataset
        if j!=4:
            X[i-1,j-4]=float(table[i,j])
            
X_transpose=np.transpose(X)

for i in range(q):
    M1=X_transpose@X         # X^T * X
    M1=np.linalg.inv(M1)    #inverse of the matrix M1
    M1=M1@X_transpose       #multiplying the matrix M1 with X transpose
    M2=np.reshape(Y[:,i],(n,1))     #Y1,Y2
    M3=M1@M2
    for j in range(p+1):
        beta[j,i]=M3[j]

Y_estimated = X@beta
#Y=XB

#-----------------------------------------------------------------------------

#plotting 

plt.plot(Y[:,0],Y[:,1],'o',color='black',label='Actual Scores')
#in the plot the black dots are represents actual scores
plt.plot(Y_estimated[:,0],Y_estimated[:,1],'o',color='red',label='Estimated Scores')
#in the plot the red dots represents estimated scores
plt.xlabel('Average Reading Scores')
plt.ylabel('Average Math Scores')
plt.title('For Visualization')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------

#error

y_error=[[0.000 for i in range(p)]for j in range(n)]
y_error=np.array(y_error)

mean_error=np.array([0,0])
for i in range(n):
    for j in range(q):
        y_error[i,j]=(Y[i,j]-Y_estimated[i,j])/Y_estimated[i,j] #actual-estimated/estimated
        y_error[i,j]=abs(y_error[i,j]*100)      #percentage
        mean_error[j]+=y_error[i,j]     #meanerror
mean_error=mean_error/n
print("The mean percentage error for read score is ",mean_error[0],"%")
print("The mean percentage error for math score is ",mean_error[1],"%")
