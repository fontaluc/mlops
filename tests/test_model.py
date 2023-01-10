import torch
from src.models.model import MyAwesomeModel
import pytest

model = MyAwesomeModel()

@pytest.mark.parametrize("batch_size", [1, 64])
def test_model(batch_size):
    input = torch.randn(batch_size, 28, 28)
    output = model(input)
    assert output.shape == torch.Size([batch_size, 10]), "Output didn't have the correct shape"

def test_error_on_wrong_shape():
    with pytest.raises(ValueError) as excinfo:
        model(torch.randn(1, 2, 3))
    assert "Expected each sample to have shape [28, 28]" in str(excinfo), "Expected exception was not raised"