import sys
import numpy as ny

#funkcja sigmoid

def nonline(x,deriv = False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1+ny.exp(-x))

#wejście
'''
 1 - 0001;5 - 0101
 2 - 0010;6 - 0110
 3 - 0011;7 - 0111
 4 - 0100;8 - 1000 
'''
X = ny.array([[0,0,0,0,0,0,0,1],
              [0,0,0,1,1,1,1,0],
              [0,1,1,0,0,1,1,0],
              [1,0,1,0,1,0,1,0]])


#wyjscie

output = input("Podaj cyfre od 1-8 ")
output = int(output)

if (output == 1):
    y = ny.array([[0,0,0,1]]).T
elif (output == 2):
    y = ny.array([[0,0,1,0]]).T
elif (output == 3):
    y = ny.array([[0,0,1,1]]).T
elif (output == 4):
    y = ny.array([[0,1,0,0]]).T
elif (output == 5):
    y = ny.array([[0,1,0,1]]).T
elif (output == 6):
    y = ny.array([[0,1,1,0]]).T
elif (output == 7):
    y = ny.array([[0,1,1,1]]).T
elif (output == 8):
    y = ny.array([[1,0,0,0]]).T

    
ny.random.seed(1)

syn0 = 2*ny.random.random((8,1)) - 1

for iter in range(10000):

    l0 = X
    l1 = nonline(ny.dot(l0,syn0))

    l1_error = y - l1
    l1_delta = l1_error * nonline(l1,True)
    syn0 += ny.dot(l0.T,l1_delta)


print (l1)    
