# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:30:45 2023

@author: 33699
"""
import torch
from tests import _PATH_DATA
import os.path
import pytest

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + '/processed/train.pt') or not os.path.exists(_PATH_DATA + '/processed/test.pt'), reason="Data files not found")
def test_data():
    train_set = torch.load(_PATH_DATA + '/processed/train.pt')
    test_set = torch.load(_PATH_DATA + '/processed/test.pt')
    N_train = 25000
    N_test = 5000
    assert train_set['images'].shape == torch.Size([N_train, 28, 28]), "Train set didn't have the correct shape"
    assert test_set['images'].shape == torch.Size([N_test, 28, 28]), "Test set didn't have the correct shape"
    assert all(i in train_set['labels'] for i in range(10)), "All labels were not represented in the train set"
    assert all(i in test_set['labels'] for i in range(10)), "All labels were not represented in the test set"