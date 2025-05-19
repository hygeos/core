from xml.etree import ElementTree as ET
from core.static import interface
from pathlib import Path
import pandas as pd


def read_xml(path: str|Path) -> dict:
    """
    Function to read xml file

    Args:
        path (str | Path): Path of xml file
    """
    
    # Recurse function
    def parse(node, dic):
        
        # Check if leaf 
        l = list(node)
        if len(l) == 0: 
            if 'name' in node.attrib: 
                attrs = node.attrib
                tag = attrs.pop('name')
                dic[tag] = attrs
            elif node.attrib: dic[node.tag] = node.attrib
            else: dic[node.tag] = node.text
            return
        
        # Update dictonary
        if 'name' in node.attrib: tag = node.attrib['name']
        else: tag = node.tag
        dic[tag] = {}
        
        # Recurse over node
        for item in l: parse(item, dic[tag])
    
    outdic = {}
    parse(ET.parse(path).getroot(), outdic)
    return outdic

def read_csv(path: str|Path, **kwargs) -> pd.DataFrame:
    """
    Function to read csv file without taking care of tabulation and whitespaces

    Args:
        path (str | Path): Path of csv file
        kwargs: Keyword arguments of `read_csv` function from pandas

    Returns:
        DataFrame: Output table in pandas DataFrame format
    """
    # verify that the csv_file exists
    csv_file = Path(path)
    assert csv_file.is_file()
    
    # open csv file
    df = pd.read_csv(csv_file, skipinitialspace=True, **kwargs)
    
    # remove trailing whitespaces
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x) 
    df.columns = [c.strip() for c in df.columns]
    return df

@interface
def select(table, where:tuple, cols:str|list = None):
    """
    Selection function in a pandas DataFrame with a condition 

    Args:
        dataframe (pd.DataFrame): Input table from which to select
        where (tuple): Condition to use for the selection
        cols (str | list): Name of the columns to return
    
    Example:
        select(df, ('col_1','=',20), ['col_2','col_3'])
    """ 
    condition = op_map[where[1]](table[where[0]], where[2])
    
    result = table[condition]
    if cols is not None: 
        result = result[cols]
    return result

@interface
def select_cell(table, where:tuple, col:str):
    """
    Function for selecting a single cell value in a pandas DataFrame with a condition

    Args:
        dataframe (pd.DataFrame): Input table from which to select
        where (tuple): Condition to use for the selection
        col (str | list): Name of the column to return
    
    Example:
        select_cell(df, ('col_1','=',20), 'col_2')
    """  
    df = select(table, where, col)
    assert len(df) == 1, f'Expected to return only one values (got {len(df)})'
    return df.values[0]


op_map = {"=": lambda a, b: a == b,
          "==": lambda a, b: a == b,
          ">": lambda a, b: a > b,
          "<": lambda a, b: a < b,
          ">=": lambda a, b: a >= b,
          "<=": lambda a, b: a <= b,
          "!=": lambda a, b: a != b,}