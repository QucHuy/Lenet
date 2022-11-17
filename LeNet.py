from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K




class LeNet:
    @staticmethod
    def build(numChannels, rows, cols, classes, activation = "relu", weightsPath = None):
        model = Sequential()
        inputShape = (rows,cols, numChannels)

        if K.image_data_format() == "channels_first":
            inputShape = (numChannels,rows,cols)

        model.add(Conv2D(6, 5 , padding ="same", input_shape= inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(Conv2D(16, 5, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation(activation))
        model.add(Dense(84))
        model.add(Activation(activation))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        if weightsPath is not None:
            model.load_weights(weightsPath)
            # return the constructed network architecture
        return model
