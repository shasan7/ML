import numpy as np

# Define the signum activation function
def signum(x):
    return np.where(x >= 0, 1, -1)

# Perceptron class
class Perceptron:
    def __init__(self, learning_rate, initial_weights):
        self.learning_rate = learning_rate
        self.weights = np.array(initial_weights)

    def predict(self, inputs):
        # Compute the linear combination of inputs and weights
        output = np.dot(inputs, self.weights)
        # Apply signum activation function
        return signum(output)

    def train(self, inputs, labels, max_epochs=1000):
        epoch = 0
        while epoch < max_epochs:
            error_vec = np.empty(len(labels))
            for i, input_vector in enumerate(inputs):
                # Get prediction for i th training pattern
                prediction = self.predict(input_vector)
                # Calculate error for i th training pattern
                error_vec[i] = labels[i] - prediction
                # Update weights if there is an error
                if error_vec[i] != 0:
                    self.weights += self.learning_rate * error_vec[i] * input_vector
            rmse = np.sqrt(np.mean(np.square(error_vec)))
            # Print status after each epoch
            print(f'Epoch {epoch+1}, RMS Error: {rmse}')
            # Stop training if RMS Error for an epoch is zero
            if rmse == 0:
                break
            epoch += 1

# User-provided values
initial_weights = [0.5, -0.02, 0.35]
learning_rate = 0.1
inputs = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]])
labels = np.array([1, -1, 1, 1])

# Create the perceptron
perceptron = Perceptron(learning_rate, initial_weights)

# Train the perceptron until the RMS Error is zero in any epoch or max epochs is reached
perceptron.train(inputs, labels)

# Final weights
print("Final weights:", perceptron.weights)