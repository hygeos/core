
def select(table, where:tuple, cols:str|list = None):
    """
    Function to select 

    Args:
        dataframe (pd.DataFrame): Input table to select in 
        where (tuple): _description_
        cols (str | list): 
    """  
    condition = op_map[where[1]](table[where[0]], where[2])
    return table[condition][cols]

def select_one(table, where:tuple, col:str = None):
    """
    Function to select 

    Args:
        dataframe (pd.DataFrame): Input table to select in 
        where (tuple): _description_
        cols (str | list): 
    """  
    df = select(table, where, col)
    assert len(df) == 1, 'Expected to return only one values'
    return df[0] 

op_map = {"=": lambda a, b: a == b,
          ">": lambda a, b: a > b,
          "<": lambda a, b: a < b,
          ">=": lambda a, b: a >= b,
          "<=": lambda a, b: a <= b,
          "!=": lambda a, b: a != b,}