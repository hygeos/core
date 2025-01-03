from core.static import interface
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as et

        
def xml_to_dict(element):
    """
    Recursively convert an XML element and its children into a nested dictionary.
    Special handling for 'neighbor' and country elements.
    """
    result = {}

    for child in element:
        if len(child) != 0: 
            country_name = child.attrib['name'] 
            country_data = xml_to_dict(child)
            result[country_name] = country_data
        elif len(child.attrib) != 0:
            info = child.attrib
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(info)
        else:
            # Handle other elements normally
            result[child.tag] = xml_to_dict(child)

    # Add attributes to the result if they exist and are not handled
    attrib = element.attrib
    if attrib:
        if 'name' in attrib: attrib.pop('name')
        result.update(attrib)

    # Add text content if no children exist
    if not result and element.text:
        return element.text.strip()

    return result

def read_xml(filepath: str | Path):
    """
    Parse an XML file and return a nested dictionary.
    """
    tree = et.parse(filepath)
    root = tree.getroot()
    return {root.tag: xml_to_dict(root)}

def read_xml_from_text(xml_string: str) -> dict:
    """
    Parse an XML string and return a nested dictionary.
    """
    root = et.fromstring(xml_string)
    return {root.tag: xml_to_dict(root)}

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
    return table[condition][cols]

@interface
def select_cell(table, where:tuple, col:str = None):
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