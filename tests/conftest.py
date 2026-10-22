'''Shared test setup: a state starts from a random tensor, so the suite fixes the generator to make
a run reproducible.'''
import pytest
import torch


@pytest.fixture(autouse=True)
def seed_generator():
    '''Seed the global torch generator before every test.'''
    torch.manual_seed(0)
