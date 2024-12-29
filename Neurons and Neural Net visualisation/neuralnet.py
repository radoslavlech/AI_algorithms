import numpy as np

LEARNING_RATE = 0.01

class Layer:
    def __init__(self,activation_function):
        self.activation_function = activation_function
        self.activation = None

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





class Neuron(Layer):
    def __init__(self,input_layer,activation_function):
        super().__init__(activation_function)
        self.input_layer = input_layer
        self.b = 0

        if self.input_layer:
            self.weights = np.array([np.random.uniform(-0.01, 0.01) for i in range(self.input_layer.n)],dtype='float64')
            self.update()
        else:
            self.weights = None

    def update(self):
        self.s = np.dot(np.array(self.input_layer.activations), self.weights)+self.b
        self.activation = self.activation_func(self.s)


class InputNeuron(Neuron):
    def __init__(self,data,activation_function):
        super().__init__(None,activation_function)
        self.X = np.array(data[:2])
        self.weights = np.array([np.random.uniform(-0.01, 0.01) for component in self.X],dtype='float64')
        self.update()

    def update(self):
        self.s = np.dot(np.array(self.X), self.weights)+self.b
        self.activation = self.activation_func(np.dot(self.weights, self.X) + self.b)





class FullyConnectedLayer(Layer):
    def __init__(self,input_layer,activation_function,n_neurons):
        super().__init__(activation_function)
        self.input_layer = input_layer
        self.n = n_neurons
        self.b = 0
        self.neurons = self.add_neurons()[0]
        self.weights = np.hstack([neuron.weights for neuron in self.neurons])
        self.update()
        self.s = np.hstack([neuron.s for neuron in self.neurons])

    def add_neurons(self):
        neurons = []
        activations = []
        for i in range(self.n):
            neuron = Neuron(self.input_layer,self.activation_function)
            neurons.append(neuron)
            activations.append(neuron.activation)
        return neurons,activations

    def update(self):
        self.activations = self.activation_func(np.dot(self.weights, self.input_layer.activations) + self.b)



class InputLayer(Layer):
    def __init__(self,data,activation_function,n_neurons):
        super().__init__(activation_function)
        self.X = data[:2]
        self.b = 0
        self.n = n_neurons
        self.neurons = self.add_neurons()[0]
        self.activations = np.array(self.add_neurons()[1])
        self.weights = np.hstack([neuron.weights for neuron in self.neurons])
        self.s = np.hstack([neuron.s for neuron in self.neurons])


    def add_neurons(self):
        neurons = []
        activations = []
        for i in range(self.n):
            neuron = InputNeuron(self.X,self.activation_function)
            neurons.append(neuron)
            activations.append(neuron.activation)
        return neurons,activations

    def update(self):
        self.activations = self.activation_func(np.dot(self.weights, self.X) + self.b)


class NeuralNet:
    def __init__(self,train_data,size,activation_function):
        #size of the form (4,3,2)
        self.size = size
        self.activation_function = activation_function
        self.train_data = train_data
        self.layers = []
        self.generate()


    def generate(self):
        input_layer = InputLayer(self.train_data[0],'sigmoid',self.size[0])

        self.layers.append(input_layer)
        for i, n in enumerate(self.size[1:]):
            layer = FullyConnectedLayer(self.layers[-1],self.activation_function,n)
            self.layers.append(layer)
        self.y = self.layers[-1].activations

    def backpropagate(self,sample):
        if sample[2] == 0:
            target = np.array([1,0])
        else:
            target = np.array([0,1])
        error = np.sum(self.y - target)

        multiplier = error*LEARNING_RATE
        for idx, layer in enumerate(reversed(self.layers)):
            print(f"multiplier: {multiplier}")
            for neuron in layer.neurons:
                multiplier * neuron.activation_derivative(neuron.s)
                delta





    def feed_forward(self):
        for layer in self.layers:
            layer.update()
            for neuron in layer.neurons:
                neuron.update()
            self.y = self.layers[-1].activations

    def train(self,epochs=10):
        for epoch in range(epochs):
            print(f'Training epoch {epoch+1}/{epochs}')
            for i,sample in enumerate(self.train_data):
                self.backpropagate(sample)
                self.feed_forward()






