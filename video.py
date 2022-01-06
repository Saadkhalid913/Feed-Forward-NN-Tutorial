import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

n = 1500

x, y = make_moons(n_samples=n, noise = 0.15)
x_train, x_test, y_train, y_test = train_test_split(x,y)

# the below comments are to plot out data

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
    # This function will take a one dimentional
    # array as input, and encode each element of 
    # that array as a one hot vector.
    # output shape is (n, num_classes)
    encoded = np.zeros((len(y), classes))
    for index, i in enumerate(y):
        encoded[index][i] = 1
    return encoded


# we perform encoding on the training and the testing set 
y_train = encode_categorical(y_train, classes=2)        
y_test = encode_categorical(y_test, classes=2)    


class NN():
    def __init__(self, input_neurons = 2, hidden_neurons=3, output_neurons = 2):

        # initializing the first layer weights and biases
        self.W1 = np.random.rand(input_neurons, hidden_neurons)
        self.B1 = np.random.rand(1, hidden_neurons)

        # initializing the second layer weights and biases
        self.W2 = np.random.rand(hidden_neurons, output_neurons)
        self.B2 = np.random.rand(1, output_neurons)

    def sigmoid(self, x: np.array, derivative = False):
        # sigmoid activation function
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

            # this is the derivative of the loss WRT A2
            DLossA2 = self.MSELoss(A2, y, derivative=True)
            # this is the derivative of A2 WRT Z2
            DLossZ2 = self.sigmoid(Z2, derivative=True)
            
            # this is the derivative of the loss WRT Z2
            # (We applied the chain rule by multiplying
            # the above derivatives)
            gradient = np.multiply(DLossA2, DLossZ2)

            # we transpose the inputs of the second layer (A1)
            # and take the dot product witht the derivative 
            # of loss WRT Z2
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

    def predict(self, x):
        Z1 = np.dot(x, self.W1) + self.B1
        A1 = self.sigmoid(Z1)

        Z2 = np.dot(A1, self.W2) + self.B2
        A2 = self.sigmoid(Z2)

        return A2

network = NN(input_neurons = 2, hidden_neurons = 3, output_neurons=2)
network.backprop(x_train, y_train, epochs=1000, learning_rate=0.01)

preds = np.argmax(network.predict(x_test), axis = 1)
y_test = np.argmax(y_test, axis = 1)

accuracy = 100 * np.sum(preds == y_test ) / len(y_test) 
print(f"Testing Accuracy: {accuracy}%")