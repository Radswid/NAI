import sys
import numpy as ny

#funkcja sigmoid

def nonline(x,deriv = False):
    if (deriv == True):
        return x*(1-x)
    return 1/(1+ny.exp(-x))

#wejście
'''
 1 - 00100 01100 10100 00100 00100 00100 00100
 2 - 00110 01001 10001 00010 00100 01000 11111
 3 - 01110 10001 00001 00110 00001 10001 01110
 
'''
X = ny.array([[0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,1,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1],
              [0,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0]])


#wyjscie

output = input("Podaj cyfre od 1-3")
output = int(output)

if (output == 1):
    y = ny.array([0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]).T
elif (output == 2):
    y = ny.array([0,0,1,1,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1]).T
elif (output == 3):
    y = ny.array([0,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0]).T

ny.random.seed(1)

syn0 = 2*ny.random.random((35,35))-1

for iter in range(10000):

    l0 = X
    l1 = nonline(ny.dot(l0,syn0))

    l1_error = y - l1
    l1_delta = l1_error * nonline(l1,True)
    syn0 += ny.dot(l0.T,l1_delta)

print (l1)    
