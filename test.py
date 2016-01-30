import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
X = np.array([[0,0,0,0,0,0,0,1],
              [0,0,0,1,1,1,1,1],
              [0,1,1,0,0,1,1,1],
              [1,0,1,0,1,0,1,1]])
                
output = input("Podaj cyfre od 1-7 ")
output = int(output)

if (output == 1):
    y = np.array([[0,0,0,1]]).T
elif (output == 2):
    y = np.array([[0,0,1,0]]).T
elif (output == 3):
    y = np.array([[0,0,1,1]]).T
elif (output == 4):
    y = np.array([[0,1,0,0]]).T
elif (output == 5):
    y = np.array([[0,1,0,1]]).T
elif (output == 6):
    y = np.array([[0,1,1,0]]).T
elif (output == 7):
    y = np.array([[0,1,1,1]]).T


np.random.seed(1)

syn0 = 2*np.random.random((8,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):

	
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    #jak bardzo się pomyliliśmy względem naszego celu
    l2_error = y - l2
    
    
    l2_delta = l2_error*nonlin(l2,deriv=True)

    #propagacja wsteczna wagi z warstwy l2 przezucone są do l1 i sprawdza się jak wartości l1 przyczyniły się do błędu  w l2
    l1_error = l2_delta.dot(syn1.T)
    
    
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
print (l2)
