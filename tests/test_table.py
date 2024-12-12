from core.table import select, select_one

import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def dataset():
    # Generation of test dataset
    shape = (20,5)
    array = np.arange(shape[0]*shape[1],dtype=int).reshape(shape)
    cols  = [f'col_{i+1}' for i in range(shape[1])] 
    return pd.DataFrame(array, columns=cols)

@pytest.mark.parametrize("cols", [
    "col_1",
    ['col_1','col_5']
])
@pytest.mark.parametrize("conditions", [("col_2", '=', 6)])
def test_select(dataset, conditions, cols):
    out = select(dataset, cols=cols, where=conditions)
    assert isinstance(out, pd.Series|pd.DataFrame)

@pytest.mark.parametrize("conditions", [("col_2", '=', 6)])
def test_select_one(dataset, conditions):
    out = select_one(dataset, col='col_1', where=conditions)
    assert isinstance(out, np.int64)