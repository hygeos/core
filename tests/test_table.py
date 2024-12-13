from core.table import *
from core.static.Exceptions import InterfaceException
from pathlib import Path

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

def test_read_csv():
    csv_file = Path(__file__).parent/'inputs'/'test.csv'
    assert csv_file.is_file()
    df = read_csv(csv_file)
    print(df)

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

@pytest.mark.parametrize("conditions", [("col_2", '=', 6)])
def test_check_input_swap(dataset, conditions):
    with pytest.raises(InterfaceException):
        out = select_one(dataset, 'col_1', conditions)