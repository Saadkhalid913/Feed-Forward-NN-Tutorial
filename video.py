import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

n = 1500

x, y = make_moons(n_samples=n)
x_train, x_test, y_train, y_test = train_test_split(x,y)

# c1 = x_train[y_train == 0]
# c2 = x_train[y_train == 1]

# plt.scatter(c1[ : , 0], c1[: , 1], c="blue")
# plt.scatter(c2[ : , 0], c2[: , 1], c="red")

# plt.show()

# [0,1,1,0]

# [[1,0]
#  [0,1]
#  [0,1]
#  [1,0]]



def encode_categorical(y, classes = 2):
    encoded = np.zeros((len(y), classes))
    for index, i in enumerate(y):
        encoded[index][i] = 1
    return encoded

y_train = encode_categorical(y_train, classes=2)        
y_test = encode_categorical(y_test, classes=2)    


class NN():
    def __init__(self, input_neurons = 2, hidden_neurons=3, output_neurons = 2):
        self.W1 = np.random.rand(input_neurons, hidden_neurons)
        self.B1 = np.random.rand(1, hidden_neurons)

        self.W2 = np.random.rand(hidden_neurons, output_neurons)
        self.B2 = np.random.rand(1, output_neurons)

    def sigmoid(self, x, derivative = False):
        if derivative:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        Z1 = np.dot(x, self.W1) + self.B1
        A1 = self.sigmoid(Z1)

        Z2 = np.dot(A1, self.W2) + self.B2
        A2 = self.sigmoid(Z2)

        return Z1, A1, Z2, A2
    
    def MSELoss(self, y_pred, y_actual, derivative = False):
        if derivative:
            return -(y_actual - y_pred)
        else:
            return np.power((y_actual - y_pred), 2)


    def backprop(self, x, y, epochs = 100, learning_rate = 0.01):
        for i in range(epochs):
            Z1, A1, Z2, A2 = self.forward(x)

            accuracy = np.sum(np.argmax(A2, axis = 1) == np.argmax(y, axis = 1)) / len(y)

            print(f"Epoch #{i + 1} -- Accuracy: {accuracy}")


            oldW2 = self.W2.copy()

            DLossA2 = self.MSELoss(A2, y, derivative=True)
            DLossZ2 = self.sigmoid(Z2, derivative=True)

            gradient = np.multiply(DLossA2, DLossZ2)

            weight_update = np.dot(A1.T, gradient)
            bias_update = np.sum(gradient, axis = 0)

            self.W2 += -1 * learning_rate * weight_update
            self.B2 += -1 * learning_rate * bias_update
            
            gradientWRTA1 = np.dot(gradient, oldW2.T)
            D_A1_WRT_Z1 = self.sigmoid(Z1, derivative=True)

            gradientWRTZ1 = np.multiply(gradientWRTA1, D_A1_WRT_Z1)

            weight_update = np.dot(x.T, gradientWRTZ1)
            bias_update = np.sum(gradientWRTZ1, axis = 0)

            self.W1 += -1 * learning_rate * weight_update
            self.B1 += -1 * learning_rate * bias_update

