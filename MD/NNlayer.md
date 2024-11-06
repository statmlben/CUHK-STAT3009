# Neural Network Layers

**Dense Layer:**

A Dense layer, also known as a fully connected layer, is a type of neural network layer where every input is connected to every output. It's the most common type of layer used in neural networks.

**General Usage and Outcome:**

```
from keras.layers import Dense

# Create a Dense layer with 64 units
dense_layer = Dense(64, input_shape=(10,))
```

In the above example, we create a Dense layer with 64 units, and the input shape is (10,). This means that the input to this layer should be a 10-dimensional vector, and the output will be a 64-dimensional vector.

**Key Arguments:**

* `units`: The number of neurons in the layer.
* `input_shape`: The shape of the input data.
* `activation`: The activation function to use (e.g. 'relu', 'sigmoid', etc.).
* `kernel_initializer`: The initializer for the kernel weights.
* `bias_initializer`: The initializer for the bias weights.

**Example:**

```
from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add a Dense layer with 64 units
model.add(Dense(64, input_shape=(10,), activation='relu'))

# Add another Dense layer with 32 units
model.add(Dense(32, activation='relu'))

# Add a final Dense layer with 10 units
model.add(Dense(10, activation='softmax'))
```

In this example, we create a Sequential model and add three Dense layers. The first layer has 64 units, the second layer has 32 units, and the final layer has 10 units. The activation function for each layer is 'relu' except for the final layer which is 'softmax'.

**Embedding Layer:**

An Embedding layer is a type of layer that converts positive integers (indexes) into dense vectors of fixed size. It's often used as the first layer in a neural network to convert categorical data into a numerical representation.

**General Usage and Outcome:**

```
from keras.layers import Embedding

# Create an Embedding layer with 128 dimensions and 10000 unique words
embedding_layer = Embedding(input_dim=10000, output_dim=128, input_length=10)
```

In the above example, we create an Embedding layer with 128 dimensions and 10000 unique words. The input length is 10, which means that the input to this layer should be a sequence of 10 integers.

**Key Arguments:**

* `input_dim`: The number of unique words in the vocabulary.
* `output_dim`: The dimensionality of the output vectors.
* `input_length`: The length of the input sequences.

**Example:**

```
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# Create a Sequential model
model = Sequential()

# Add an Embedding layer with 128 dimensions and 10000 unique words
model.add(Embedding(input_dim=10000, output_dim=128, input_length=10))

# Add a Flatten layer to flatten the output
model.add(Flatten())

# Add a Dense layer with 64 units
model.add(Dense(64, activation='relu'))

# Add a final Dense layer with 10 units
model.add(Dense(10, activation='softmax'))
```

In this example, we create a Sequential model and add an Embedding layer with 128 dimensions and 10000 unique words. We then add a Flatten layer to flatten the output, followed by two Dense layers. The first Dense layer has 64 units, and the final layer has 10 units.