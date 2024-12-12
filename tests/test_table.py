from core.table import select, select_one

import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def dataset():
    # Generation of test dataset
    shape = (20,5)
    array = np.linspace(1,shape[0]*shape[1]).reshape(shape)
    cols  = [f'col_{i+1}' for i in range(shape[1])] 
    return pd.DataFrame(array, columns=cols)

@pytest.skip()
@pytest.mark.parametrize("cols", [
    "col_1",
    ['col_1','col_5']
])
@pytest.mark.parametrize("conditions", [
    [("col_2", '=', 20)],
    [("col_2", '=', 20),("col_3", '=', 20)]
])
def test_select(dataset, conditions, cols):
    out = select(dataset, cols=cols, where=conditions)

@pytest.skip()
@pytest.mark.parametrize("conditions", [
    [("col_2", '=', 20)],
    [("col_2", '=', 20),("col_3", '=', 20)]
])
def test_select_one(dataset, conditions):
    out = select_one(dataset, col='col_1', where=conditions)