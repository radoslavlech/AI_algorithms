    def backpropagate(self,sample):
        if sample[2] == 0:
            target = np.array([1,0])
        else:
            target = np.array([0,1])
        error = np.sum(self.y - target)

        multiplier = error*LEARNING_RATE*self.layers[-1].activation_derivative(self.layers[-1].s)
        multiplier = multiplier.transpose()[0]

        for layer_idx in range(len(self.size)-1,0,-1):
            print(f"Updating weights of {layer_idx}. layer")
            print(f"multiplier: {multiplier}")
            print(f"activations: {self.layers[layer_idx-1].activations}")
            print(f"weights: {self.layers[layer_idx].weights}")
            delta_w = np.zeros_like(self.layers[layer_idx].weights)


            for j, row in enumerate(multiplier):
                delta_w[j] = float(row)*self.layers[layer_idx-1].activations
            print(f"delta_w: {delta_w}")

            weights_dot_derivative = np.dot(self.layers[layer_idx].weights,self.layers[layer_idx-1].activation_derivative(self.layers[layer_idx-1].s))
            print(f"weights_dot_derivative: {weights_dot_derivative}")
            self.layers[layer_idx].weights += delta_w

            new_multiplier = np.zeros((len(weights_dot_derivative),len(multiplier)))
            print(new_multiplier)
            for i, row in enumerate(new_multiplier):
                for j, col in enumerate(row):
                    new_multiplier[i,j] = weights_dot_derivative[i]*multiplier[j]

            multiplier = new_multiplier
