from typing import Tuple

import numpy
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from sklearn.preprocessing import OneHotEncoder

from fragile.core import Swarm, HistoryTree
from fragile.core.utils import random_state


class ConvolutionalNeuralNetwork:
    """
    Convolutional neural network build with Keras to fit stacked images with only \
    one channel.

    It is meant to be used as a Model for imitation learning problems with \
    discrete action spaces.
    """

    def __new__(cls, input_shape: Tuple[int, int, int], n_actions: int):
        """
        Return the instantiated Keras model.

        Args:
            input_shape: (n_stacked_frames, frame_width, frame_height)
            n_actions: Number of discrete actions to be predicted.

        """
        model = Sequential()
        model.add(
            Conv2D(
                32,
                8,
                strides=(4, 4),
                padding="valid",
                activation="relu",
                input_shape=input_shape,

            )
        )
        model.add(
            Conv2D(
                64,
                4,
                strides=(2, 2),
                padding="valid",
                activation="relu",
                input_shape=input_shape,
            )
        )
        model.add(
            Conv2D(
                64,
                3,
                strides=(1, 1),
                padding="valid",
                activation="relu",
                input_shape=input_shape,
            )
        )
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dense(n_actions))
        model.compile(
            loss="mean_squared_error",
            optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
            metrics=["accuracy"],
        )
        model.summary()
        return model


class ModelTrainer:
    def __init__(self, input_shape, n_actions):
        classes = numpy.arange(n_actions).reshape(-1, 1)
        self.oh_encoder = OneHotEncoder(sparse=False).fit(classes)
        self.action_space = n_actions
        self.model = ConvolutionalNeuralNetwork(input_shape, n_actions)

    def move(self, state):
        actions = self.model.predict(
            numpy.expand_dims(numpy.asarray(state).astype(numpy.float64), axis=0), batch_size=1
        )
        return numpy.argmax(actions[0])

    def train(self, swarm: Swarm, batch_size=32, epochs: int=500, verbose: int=0):
        random_batches = swarm.tree.iterate_nodes_at_random(batch_size=batch_size,
                                                            names=['observs', "actions"])
        for observs, actions in random_batches:
            actions = self.oh_encoder.transform(actions.reshape(-1, 1))
            self.model.fit(
                observs,
                actions,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
            )
