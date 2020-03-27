from typing import Callable, Tuple

import numpy
from plangym import AtariEnvironment, ClassicControl
import pytest

from fragile.core.bounds import Bounds
from fragile.core.env import DiscreteEnv, Environment
from fragile.core.states import StatesEnv, StatesModel
from fragile.optimize.env import Function

N_WALKERS = 10


def discrete_atari_env():
    env = AtariEnvironment(name="MsPacman-v0", clone_seeds=True, autoreset=True)
    env.reset()
    env = DiscreteEnv(env)
    return env


def classic_control_env():
    env = ClassicControl("CartPole-v0")
    env.reset()
    env = DiscreteEnv(env)
    return env


def create_env_and_model_states(name="classic") -> Callable:
    def _atari_env():
        env = discrete_atari_env()
        params = {"actions": {"dtype": numpy.int64}, "critic": {"dtype": numpy.float32}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        states.update(actions=numpy.ones(N_WALKERS), critic=numpy.ones(N_WALKERS))
        return env, states

    def _classic_control_env():
        env = classic_control_env()
        params = {"actions": {"dtype": numpy.int64}, "dt": {"dtype": numpy.float32}}
        states = StatesModel(state_dict=params, batch_size=N_WALKERS)
        states.update(actions=numpy.ones(N_WALKERS), dt=numpy.ones(N_WALKERS))
        return env, states

    if name.lower() == "pacman":
        return _atari_env
    else:
        return _classic_control_env


env_fixture_params = ["classic", "pacman"]


@pytest.fixture(params=env_fixture_params, scope="class")
def env_data(request) -> Tuple[Environment, StatesModel]:
    if request.param in env_fixture_params:
        env, model_states = create_env_and_model_states(request.param)()

    else:
        raise ValueError("Environment not well defined: %s" % request.param)
    return env, model_states


@pytest.fixture(scope="class")
def batch_size():
    return 10


class TestEnvironment:
    def test_repr(self, env_data):
        env, model_states = env_data
        repr_env = env.__repr__()
        assert isinstance(repr_env, str)

    def test_reset(self, env_data):
        env, model_states = env_data
        states = env.reset()
        assert isinstance(states, env.STATE_CLASS), states

        batch_size = 10
        states = env.reset(batch_size=batch_size)
        assert isinstance(states, env.STATE_CLASS), states
        for name in states.keys():
            assert states[name].shape[0] == batch_size

    def test_get_params_dir(self, env_data):
        env, model_states = env_data
        params_dict = env.get_params_dict()
        assert isinstance(params_dict, dict)
        for k, v in params_dict.items():
            assert isinstance(k, str)
            assert isinstance(v, dict)
            for ki in v.keys():
                assert isinstance(ki, str)

    def test_step(self, env_data, batch_size):
        env, model_states = env_data
        states = env.reset(batch_size=batch_size)
        new_states = env.step(model_states=model_states, env_states=states)
        assert new_states.oobs.sum() == 0
        assert new_states.terminals.sum() == 0

    def test_state_shape(self, env_data):
        env, model_states = env_data
        assert hasattr(env, "states_shape")
        assert isinstance(env.states_shape, tuple)

    def test_observs_shape(self, env_data):
        env, model_states = env_data
        assert hasattr(env, "observs_shape")
        assert isinstance(env.observs_shape, tuple)

    def test_states_from_data(self, env_data, batch_size):
        env, model_states = env_data
        states = numpy.zeros((batch_size, 5)).tolist()
        observs = numpy.ones((batch_size, 5)).tolist()
        rewards = numpy.arange(batch_size).tolist()
        oobs = numpy.zeros(batch_size, dtype=bool).tolist()
        state = env.states_from_data(
            batch_size=batch_size, states=states, observs=observs, rewards=rewards, oobs=oobs
        )
        assert isinstance(state, StatesEnv)
        for val in state.vals():
            assert isinstance(val, numpy.ndarray)
            assert len(val) == batch_size
