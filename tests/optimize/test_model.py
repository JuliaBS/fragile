import pytest

from fragile.core import Bounds
from fragile.core.states import StatesEnv, StatesModel
from fragile.optimize.models import ESModel

from tests.core.test_model import TestModel

BATCH_SIZE = 10


def create_model(name="es_model"):
    if name == "es_model":
        bs = Bounds(low=-10, high=10, shape=(BATCH_SIZE,))
        return lambda: ESModel(bounds=bs)
    raise ValueError("Invalid param `name`.")


model_fixture_params = ["es_model"]


@pytest.fixture(scope="class", params=model_fixture_params)
def model(request):
    return create_model(request.param)()


@pytest.fixture(scope="class")
def batch_size():
    return BATCH_SIZE


class TestESModel:
    def create_model_states(self, model, batch_size: int = None):
        return StatesModel(batch_size=batch_size, state_dict=model.get_params_dict())

    def create_env_states(self, model, batch_size: int = None):
        return StatesEnv(batch_size=batch_size, state_dict=model.get_params_dict())

    def test_run_for_1000_predictions(self, model, batch_size):
        for _ in range(100):
            TestModel.test_predict(self, model, batch_size)
