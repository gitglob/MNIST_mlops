import sys
import pytest
from torch import rand, randn
from tests import _PATH_SRC_MODELS

sys.path.append(_PATH_SRC_MODELS)
from model import MyModel


# test shape of model output for multiple latent/output dimension combinations
@pytest.mark.parametrize("latent_dim, output_dim", [(128, 10), (64, 10), (2, 10)])
def test_model(latent_dim, output_dim):
    model = MyModel(None, latent_dim, output_dim)

    x = rand(1, 1, 28, 28)

    assert model(x).shape == (1, 10), "The output of the model is not (1, 10)"


def test_error_on_wrong_shape():
    model = MyModel(None, 128, 10)
    # test that shape errors give the right feedback
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(randn(1, 2, 3))
