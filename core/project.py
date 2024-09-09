from pathlib import Path
from typing import Optional, Any

import toml

from core.static import interface

"""
Generic project configuration module.
Simpler version of core.config module, more generic
Config path need to be passed as argument

Use example:

cfg = Config(__file__.parent / "project-config.toml")
cfg.get("key", default=-1)

"""


class Config:
    @interface
    def __init__(self, config_path: Path, section=None):
        """
        Read and initialize the configuration provided
        """

        if not config_path.is_file() and config_path.name.endswith(".toml"):
            raise RuntimeError(f"Invalid config file path {config_path}")
        
        # read config
        self.config_file = config_path
        self.config = toml.load(self.config_file)

        if section is not None:
            sections = section.split('.')
            
            # find subsection
            for sub in sections:
                if sub in self.config:
                    self.config = self.config[sub]
                else:
                    raise KeyError(f"Missing section [{section}] in file {config_path}")

    def get(
        self,
        key: str,
        section: str = None,
        *,
        default=None,
    ) -> Any:
        """
        Returns the key if present in the config file.
        If not present and no default provided: raises an error
        if default is provided (different than None): return default instead
        
        auto cast values to Path objects if its key starts with 'dir_' and its type is str
        """

        config_path = self.config_file

        # Base section
        if section is None:
            current_config = self.config # get config content
        
        # Section or nested subsection
        else: 
            sections = section.split('.')
            current_config = self.config
            
            # find subsection
            for sub in sections:
                if sub in current_config:
                    current_config = current_config[sub]
                else:
                    raise KeyError(f"Missing section [{section}] in file {config_path}")
            
        # key managing
        if key not in current_config:
            if default is not None:
                return default
            else:
                mess = "root section" if section is None else f"section [{section}]"
                raise KeyError( f"Missing key {key} in {mess} of "
                    f"file {config_path} and no defaults is provided"
                )
                
        # get value
        value = current_config[key]
        
        # auto convert value to path if its key starts with 'dir_'
        if type(value) is str and key.startswith("dir_"):
            value = Path(value)

        return value 
    
    @interface
    def ingest(self, cfg: dict, section: str):
        """
        Injects a dictionnary as a subsection config
        Usefull to merge differents configs and to propagate them easily
        """
        
        config = self.config
        sections = section.split('.')
        last_section = sections.pop(-1) # pop last element
        
        for sub in sections: # iterate in-between subsections
            if not sub in config:
                config[sub] = {}        # instantiate empty dictionnary
                config = config[sub]    # go one level deeper
        
        if last_section in self.config:
            raise RuntimeError(f"Section '{section}' already in config. Cannot ingest.")

        config[last_section] = cfg # append passed config to deepest section 
        
        return