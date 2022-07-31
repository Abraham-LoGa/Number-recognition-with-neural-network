import matplotlib.pyplot as plt
import numpy as np
train_data = np.loadtxt("mnist_test.csv", delimiter=",")
print(np.shape(train_data))
data=np.array(train_data)
row=784 # De acuerdo al shape
col=8920  # De acuerdo al shape
recolector=np.zeros((row,col))
aux=0

for k in range (10):
  for i in range (col):
    if data[i,0]==k:
      for j in range (row):
        recolector[j,aux]=data[i,j]
      aux=aux+1
P=np.array(recolector)
print(P.shape)

Z = np.vstack([P,np.ones((1,8920))])


#Valores esperados 
T = np.vstack([np.ones((1,892)),-np.ones((9,892))])



for i in range(1,10):
      # hstack es para columnas 
    T = np.hstack([T,np.vstack([-np.ones((i,892)),np.ones((1,892)),-np.ones((9-i,892))])])


  # Red neuronal adaline

R = np.dot(Z,Z.T)/8920
H = np.dot(Z,T.T)/8920
X = np.linalg.pinv(R)@H

W = X[0:784,:].T
b = X[784,:].reshape(-1,1)

index = np.ones((1,8920))
neurona_sal = np.ones((1,8920))

for q in range(8920):
    a = np.dot(W,P[:,q]).reshape(-1,1)+b
    neurona_sal[:,q] = np.amax(a)
    posicion = np.where(a==neurona_sal[:,q])
    index[:,q]=posicion[0]

    
# Valores reales
y = np.zeros((1,892))
for j in range(1,10):
    y = np.hstack([y,j*np.ones((1,892))])

numero_aciertos = np.sum(y==index)
porcentaje_aciertos = (numero_aciertos/8920)*100
print("Total Numeros identificado: "+str (numero_aciertos))
print("Porcentaje de aciertos : "+str(porcentaje_aciertos)+"%")


for k in range(10):
    indice = np.round((8919*np.random.rand(1)+1),0)
    print(indice)
    numero_reconocido = index[:,int(indice)]
    print(numero_reconocido)
    pixels = P[:,int(indice)].reshape(28,28)
    plt.imshow(pixels,cmap='gray')
    plt.title('Numero Reconocido:'+str(int(numero_reconocido)))
    plt.show()