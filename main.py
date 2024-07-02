import NeuralNetwork #My library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

##Showing data
# data = pd.read_csv("Datasets/binary_classification_data.csv")
# data = pd.read_csv("Datasets/data_training_logistic.csv")
X, Y = make_circles(1000, factor=.5, noise=.05) #Testing with circles 
# X = np.array(data[["X1", "X2"]])
# Y = np.array(data["type"])

topology = [2,4,8,1]
model = NeuralNetwork.NeuralNetwork(topology)

Y_ = Y
Y = Y.reshape(-1,1)
model.fit(X,Y, epochs= 50000)

Y_pred = model(X)

for i in range(len(X)):
    plt.scatter(X[i,0], X[i,1], color="green" if Y[i] == 1 else "red", edgecolor='k')

x_min, x_max = X[:,0].min() - 1, X[:,1].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

#inference over meshgrid
grid_points = np.c_[xx.ravel(), yy.ravel()]
predictions = model(grid_points)
predictions = predictions.reshape(xx.shape)

#generate the color
plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap='RdYlGn', alpha=0.6)


plt.show()



