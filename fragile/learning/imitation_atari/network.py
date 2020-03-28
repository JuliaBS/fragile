from typing import Tuple

import numpy
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from sklearn.preprocessing import OneHotEncoder
from plangym.wrappers import Monitor

from fragile.core import Swarm, HistoryTree
from fragile.core.utils import random_state, get_plangym_env


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
                32, 8, strides=(4, 4), padding="valid", activation="relu", input_shape=input_shape,
            )
        )
        model.add(
            Conv2D(
                64, 4, strides=(2, 2), padding="valid", activation="relu", input_shape=input_shape,
            )
        )
        model.add(
            Conv2D(
                64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=input_shape,
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
        self.obs_memory = None
        self.action_memory = None

    def memorize(self, swarm):
        observs, actions = next(
            swarm.tree.iterate_branch(swarm.best_id, batch_size=-1, names=["observs", "actions"])
        )
        self.obs_memory = (
            observs if self.obs_memory is None else numpy.vstack([observs, self.obs_memory])
        )

        actions = self.oh_encoder.transform(actions.reshape(-1, 1).astype(numpy.float64))
        self.action_memory = (
            actions
            if self.action_memory is None
            else numpy.vstack([actions, self.action_memory])
        )

    def predict(self, observ):
        actions = self.model.predict(
            numpy.expand_dims(numpy.asarray(observ).astype(numpy.float64), axis=0), batch_size=1
        )
        return numpy.argmax(actions[0])

    def evaluate(self, swarm) -> float:
        plangy_env = get_plangym_env(swarm).clone()
        plangy_env.wrap(Monitor, './video', force=True)
        terminal = False
        current_obs = plangy_env.reset(return_state=False)
        # TODO: Find out why the monitor sometimes freezes when setting a new state
        plangy_env.set_state(swarm.init_state)
        score = 0
        while not terminal:
            action = self.predict(current_obs)
            next_obs, reward, terminal, _ = plangy_env.step(action)
            score += reward
            current_obs = next_obs
        plangy_env.close()
        return score

    def train(self, batch_size=32, epochs: int = 500, verbose: int = 0):
        # index in case you want to fit only one batch
        # ix = random_state.choice(numpy.arange(len(self.action_memory)), batch_size)
        self.model.fit(
            self.obs_memory,
            self.action_memory,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        metrics = self.model.evaluate(self.obs_memory, self.action_memory, batch_size=batch_size)
        return dict(zip(self.model.metrics_names, metrics))
