import numpy as np
import numdifftools as nd

#for 3: heaviside, logistic
#for 4: sin,tanh
#for 5: sign, ReLU, Leaky ReLu



class Neuron:
    def __init__(self,input,activation_function,weights=np.array([np.random.uniform(-0.01, 0.01),np.random.uniform(-0.01, 0.01)],dtype='float64')):
        self.activation_function = activation_function
        self.X = input
        self.w = weights
        self.b=0
        self.s = np.dot(np.array(self.X[0][:2]),   self.w)+self.b
        self.a = self.activation_func(self.s)

    def activation_func(self, s):
        if self.activation_function == 'heaviside':
            return np.heaviside(s, 1)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-s))
        elif self.activation_function == 'sin':
            return np.sin(s)
        elif self.activation_function == 'tanh':
            return np.tanh(s)
        elif self.activation_function == 'relu':
            return np.maximum(0, s)
        elif self.activation_function == 'leaky_relu':
            return np.where(s > 0, s, s * 0.01)
        elif self.activation_function == 'sgn':
            return np.sign(s)

    def activation_derivative(self, s):
        if self.activation_function == 'heaviside':
            return 1
        elif self.activation_function == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-s))
            return sigmoid * (1 - sigmoid)
        elif self.activation_function == 'sin':
            return np.cos(s)
        elif self.activation_function == 'tanh':
            return 1 - np.tanh(s) ** 2
        elif self.activation_function == 'relu':
            return np.where(s > 0, 1, 0)
        elif self.activation_function == 'leaky_relu':
            return np.where(s > 0, 1, 0.01)
        elif self.activation_function == 'sgn':
            return 1


    def learning_rate(self,n,n_max,min=0.0001,max=0.01):
        return min + (max-min)*(1+np.cos(n/n_max*np.pi))

    def learn(self, epochs=10):
        n_max = (epochs-1)*(len(self.X)-1)
        for epoch in range(epochs):
            for i in range(1,len(self.X)):
                x = np.array(self.X[i][:2])
                label = self.X[i][-1]
                error = label - self.a

                self.s = np.dot(x, self.w) + self.b
                self.a = self.activation_func(self.s)

                df = self.activation_derivative(self.s)
                self.w += self.learning_rate(epoch*len(self.X)+i,n_max) * error * df * x
                self.b += self.learning_rate(epoch*len(self.X)+i,n_max) * error * df

    def classify(self, input):
        s = np.dot(np.array(input),self.w)+self.b
        return self.activation_func(s)




