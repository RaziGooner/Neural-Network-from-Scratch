import numpy as np


class Layer:
    def __init__(self):
        self.layer_input = None
        self.layer_output = None

    def forward_propagation(self, input):
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass


class FCLayer(Layer):
    def __init__(self, input_size, output_size, name='Dense Layer'):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.name = name

    def forward_propagation(self, layer_input):
        self.layer_input = layer_input
        self.layer_output = np.dot(self.weights, self.layer_input) + self.bias

        return self.layer_output

    def backward_propagation(self, output_error, leraning_rate):
        weights_error = np.dot(output_error, self.layer_input.T)
        self.weights -= (leraning_rate * weights_error)

        bias_error = output_error
        self.bias -= (leraning_rate * bias_error)

        input_error = np.dot(self.weights.T, output_error)

        return input_error


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime, name='Activation Layer'):
        self.activation_func = activation
        self.activation_func_derivative = activation_prime
        self.name = name

    def forward_propagation(self, layer_input):
        self.layer_input = layer_input
        self.layer_output = self.activation_func(self.layer_input)

        return self.layer_output

    def backward_propagation(self, output_error, leraning_rate):
        activation_prime = self.activation_func_derivative(self.layer_input)
        input_error = np.multiply(output_error, activation_prime)

        return input_error

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        s = ActivationLayer.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


class Network:
    def __init__(self):
        self.layers = []
        self.error = 0
        self.train_loss = []
        self.validation_loss = []

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, X, Y, epochs, learning_rate, X_val=None, Y_val=None, propogation_op=False):
        for epoch in range(epochs):
            total_error = 0
            total_val_error = 0
            for input, y in zip(X, Y):
                output = input
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                    if propogation_op == True:
                        print(f'Forward propogation output for {layer.name}: \n{output}\n')
                total_error += mse(y, output)

                grad = mse_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward_propagation(grad, learning_rate)
                    if propogation_op == True and isinstance(layer, FCLayer):
                        print(
                            f'Backpropagation output for {layer.name}:\n Weights: {layer.weights}\n Bias:{layer.bias[0]}\n')

                if X_val is not None and Y_val is not None:
                    for input, y in zip(X_val, Y_val):
                        output = input
                        for layer in self.layers:
                            output = layer.forward_propagation(output)
                        total_val_error += mse(y, output)

            print(f'Epoch {epoch + 1}/{epochs} completed. Training error: {total_error}')

            self.validation_loss.append(total_val_error)
            self.train_loss.append(total_error)

        print('\n')
        print(f'{epoch + 1}/{epochs} epochs completed')

    def predict(self, X):
        result = []
        for x in X:
            output = x
            for layer in self.layers:
                output = layer.forward_propagation(output)
            if output >= 0.5:
                result.append(1)
            else:
                result.append(0)
        output_str = "Predictions:\n" + "\n".join(
            [f"Input: {X[i].flatten()} => Prediction: {result[i]}" for i in range(len(X))])

        return print(output_str)
