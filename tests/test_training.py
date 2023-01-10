import torch
from torch import nn, optim
from src.models.model import MyAwesomeModel


def test_training():
    '''
    Check for parameter update during training
    Parameter is deemed updated if the gradient is not None and has non-zero value
    '''
    inputs = torch.randn(64, 28, 28)
    targets = torch.zeros(64).type(torch.LongTensor)
    model = MyAwesomeModel().float()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    log_ps = model(inputs)
    loss = criterion(log_ps, targets)
    loss.backward()
    optimizer.step()
    assert all((param.grad is not None) and (not torch.all(param.grad == 0)) for param in model.parameters())