from pathlib import Path
from typing import Optional, Any

import toml

from core.static import interface
from core import log


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
    def __init__(self, config_path: Path, allow_home_config: bool = False):
        """ Read and initialize the toml configuration file as a dictionnary0

        Args:
            config_path (Path): path of the toml config file to load
            section (str, optional): Only read section as root if provided. Defaults to None.

        Raises:
            RuntimeError: _description_
            KeyError: _description_
        """        

        if not config_path.name.endswith(".toml"):
            raise RuntimeError(f"Invalid config file extension {config_path}: expected toml")

        if not config_path.is_file():
            raise RuntimeError(f"Invalid config file path {config_path}")
        
        if allow_home_config is True:
            
            base_config_path = Path.home() / ".config"/ config_path.name
            if base_config_path.is_file():
                
                self.cfg_dict = Config.load_toml_as_dict(base_config_path)
                self.cfg_dict.update(Config.load_toml_as_dict(config_path))
                log.debug(f"Overriding home config {base_config_path} with {config_path}")
        else:
            self.cfg_dict = Config.load_toml_as_dict(config_path)



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

        # Base section
        if section is None:
            current_config = self.cfg_dict # get config content
        
        # Section or nested subsection
        else: 
            sections = section.split('.')
            current_config = self.cfg_dict
            
            # find subsection
            for sub in sections:
                if sub in current_config:
                    current_config = current_config[sub]
                else:
                    raise KeyError(f"Missing section [{section}] in config file")
            
        # key managing
        if key not in current_config:
            if default is not None:
                return default
            else:
                mess = "root section" if section is None else f"section [{section}]"
                raise KeyError( f"Missing key {key} in {mess} of in config file")
                
        # get value
        value = current_config[key]
        
        # auto convert value to path if its key starts with 'dir_'
        if type(value) is str and key.startswith("dir_"):
            value = Path(value)

        return value 
    
    @interface
    def load_toml_as_dict(toml_path: Path):
        
        # read config
        assert toml_path.is_file()
        dictio = toml.load(toml_path)
        
        return dictio