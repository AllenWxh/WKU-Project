import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD

# Load MNIST Dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalization Image Data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Create the model
def create_model(activation='relu', optimizer='adam'):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation=activation, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the performance of the model
def train_and_evaluate_model(model, epochs=5, verbose=0):
    model.fit(train_images, train_labels, epochs=epochs, verbose=verbose)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    return test_loss, test_acc

# Using different activation functions
activations = ['relu', 'sigmoid', 'tanh']
for activation in activations:
    model = create_model(activation=activation)
    loss, acc = train_and_evaluate_model(model)
    print(f"Activation: {activation} - Test accuracy: {acc}, Test loss: {loss}")

# Using different optimizers, and learning rates.
optimizers = {
    'adam': Adam(learning_rate=0.001),
    'sgd': SGD(learning_rate=0.01)
}
for opt_name, optimizer in optimizers.items():
    model = create_model(optimizer=optimizer)
    loss, acc = train_and_evaluate_model(model)
    print(f"Optimizer: {opt_name} - Test accuracy: {acc}, Test loss: {loss}")
