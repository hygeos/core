# standard library imports
from textwrap import dedent
from pathlib import Path
import argparse
import tempfile
import shutil
import os

# third party imports
import toml
        
# sub package imports
from core.output import rgb, error, disp
from core.static import interface




config_file_name = "core-config.toml"

def _load(config_path: Path=None):
    
    local_config_file = Path.cwd() / config_file_name
    global_config_file = Path.home() / config_file_name
    
    config_file = None
    
    # when file path is overriden (usefull for tests)
    if config_path is not None:
        if config_path.is_dir():
            config_path = config_path / config_file_name
        
        if not config_path.is_file():
            raise FileNotFoundError(f"Could not find {config_file_name} file at {config_path}")
        local_config_file = config_path
    
    # default behavior
    if local_config_file.is_file():
        config_file = local_config_file
    elif global_config_file.is_file():
        config_file = global_config_file
    else:
        raise FileNotFoundError(f"Could not find {config_file_name} file.\n" +
                                " -> Please initiate it with the command:\n" +
                                "   core-config-init")
    
    # read
    return toml.load(config_file), config_file


@interface
def _write_key(config_file: Path, section: str, key: str, value, comment: str = ""):
        
    section = dedent(section.strip())
    key = dedent(key.strip())
    value = dedent(value.strip())
    comment = dedent(comment.strip())
    
    if not comment.startswith('#') and len(comment) > 0:
        comment = f"# {comment} "

    found_section = False
    wait_for_next_line = False
    
    if type(value) is str:
        if "\n" in value:
            if not value.startswith('\n'):
                value = '\n' + value
            if not value.endswith('\n'):
                value = value + '\n'
            value =  '\"\"' + value + '\"\"'
        value =  f"\"{value}\""
    

    new_config_file = config_file
    config_file = config_file.rename(config_file.parent / (str(config_file.name) + ".old"))

    # Open the input file for reading and a new file for writing
    with open(config_file, 'r') as infile, open(new_config_file, 'w+') as outfile:
        
            # helpers functions
        def write_section():
                outfile.write(f"\n[{section}] # (inserted by core.config)\n")
        def write_key():
                outfile.write(f"{key} = {value} {comment}# (inserted by core.config)\n")
            
        
        for line in infile:
            
            if wait_for_next_line:
                write_key()
                wait_for_next_line = False
            
            if f"[{section}]" in line:
                wait_for_next_line = True
                found_section = True
                
            # Write the modified or unmodified line to the output file
            outfile.write(line)
        
        if not found_section:
            write_section() 
            write_key()
            
        if wait_for_next_line:
            write_key()
            
    # Path(config_file).unlink()

    os.system(f"cat {new_config_file}")


def get(section, key, *, default=None, write: bool=False, comment: str=""):
    """
    Returns the key if present in coreconfig.toml
    """
    
    loaded_config, config_path = _load()
    
    print(loaded_config)
    
    if section not in loaded_config and not write:
        raise KeyError(f"Missing section [{section}] in file {config_path}")
    
    if key not in loaded_config[section]:
        if default is None:
            raise KeyError(f"Missing key {key} in section [{section}] in file {config_path} and no defaults provided")
        else:
            if write:
                _write_key(config_path, section, key, default, comment)
            return default
    
    return loaded_config[section][key]
    
@interface
def init(install_folder: Path|str):
    
    # str to Path
    install_folder = Path(install_folder)
    
    if not install_folder.is_dir():
        error(f"Invalid path argument: {install_folder}", rgb.orange," Please provide a valid folder path")
    
    config_file = install_folder / config_file_name
    
    if config_file.is_file():
        error(rgb.orange, f'Configuration file {config_file} already exists')
        
    else:
        # Initialize default config file
        content = dedent("""
            # HYGEOS genral configuration file
            # This file has been generated by core.config.init

            [general]
            # base_dir = "PATH/TO/DATA_DIR"

            [harp]
            # data_dir = "PATH/TO/DATA_DIR/ancillary"

            [samples]
            # data_dir = "PATH/TO/DATA_DIR/sample_products"

            # EOF
            """
        )
        
        with open(config_file, "w") as fp:
            disp(rgb.blue, "Created ", rgb.default, config_file.name)
            fp.writelines(content)


def init_cmd(args=None):
    
    parser = argparse.ArgumentParser(description='Command to instantiate a default core-config.toml file in the working directory')
    parser.add_argument('path', help="path where to create config file, defaults to home folder", nargs='?', default=Path.home())
    parser.add_argument('--from-home',  action="store_true", help="Use the core-config.toml file from the home folder as template for the init")
    args = parser.parse_args()
    
    dest = Path.cwd() / config_file_name
    
    if args.from_home:
        source = Path.home() / config_file_name
        
        if not source.is_file():
            error(f"Error: could not find file {source}")    
        
        shutil.copy(source, dest)
        disp(rgb.blue, "Created file ", rgb.default, str(dest.name))
    else:
        init(Path(args.path))
    
    # simplify path if file is in current directory
    if Path(args.path).resolve() == Path.cwd():
        dest = Path(f"./{config_file_name}") 
    
    disp(rgb.orange,   "You can configure it using the command:")
    disp(rgb.default, f"    code {dest}")
    