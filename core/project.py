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
    def __init__(self, config_path: Path):
        """
        Read and initialize the configuration provided
        """

        if not config_path.is_file() and config_path.name.endswith(".toml"):
            raise RuntimeError(f"Invalid config file path {config_path}")
        
        # read config
        self.config_file = config_path
        self.config = toml.load(self.config_file)

    def get(
        self,
        key: str,
        section: Optional[str] = None,
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

        # optional section managing
        if section is None:
            current_config = self.config # get config content
        else:
            if section not in self.config:
                raise KeyError(f"Missing section [{section}] in file {config_path}")
            current_config = self.config[section] # get config section content
            
        # key managing
        if key not in current_config:
            if default is not None:
                return default
            else:
                raise KeyError( f"Missing key {key} in section [{section}] of "
                    f"file {config_path} and no defaults provided"
                )
                
        # get value
        value = current_config[key]
        
        # auto convert value to path if its key starts with 'dir_'
        if type(value) is str and key.startswith("dir_"):
            value = Path(value)

        return value 