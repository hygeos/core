import os
from pathlib import Path
from textwrap import dedent
from typing import Literal, Optional, Any

import click
import toml

from core.output import error, rgb
from core.static import interface

"""
General configuration module located in file `core-config.toml`, either in local
directory or in ~/.config/
This configuration file is designed to store common configuration values for Hygeos
software.

Typical use:
    from core import config
    dir_ancillary = config.get('dir_ancillary')
"""


config_file_name = "core-config.toml"

configfilenotfound = FileNotFoundError(
    f"Could not find {config_file_name} file.\n"
    + " -> Please initiate it with the command:\n"
    + "    python -m core.config <base_dir>"
)


def get_config_file(mode: Literal["global", "local", "auto"]) -> Path:
    """
    Returns the config file name

    Args:
        mode ('global', 'local', 'auto'). If 'auto', returns the first existing file
            starting with local then global. If none exist, returns an error.

    Returns:
        Path: path to the config file
    """

    local_config_file = Path.cwd() / config_file_name
    global_config_file = Path.home() / ".config" / config_file_name

    global_config_file_old = Path.home() / config_file_name
    if global_config_file_old.exists():
        raise DeprecationWarning(
            "global core-config file has moved from "
            f"{global_config_file_old} to {global_config_file}"
        )

    # default behavior
    if mode == "global":
        return global_config_file
    elif mode == "local":
        return local_config_file
    elif mode == "auto":
        if local_config_file.is_file():
            return local_config_file
        elif global_config_file.is_file():
            return global_config_file
        else:
            raise configfilenotfound


class Config:
    def __init__(self):
        f"""
        Read and initialize the configuration provided in {config_file_name}
        """

        # read config
        self.config_file = get_config_file("auto")
        self.config = toml.load(self.config_file)

        # Initialize directories in the root section
        for k in self.config:
            if k.startswith("dir_"):
                self.config[k] = Path(self.config[k])

                if k == "dir_data":
                    # check that dir_data is defined as absolute
                    assert self.config[k].is_absolute()
                    if not self.config[k].exists():
                        raise NotADirectoryError(
                            f"base_dir {self.config[k]} does not exist. Please create it."
                        )
                else:
                    # all other directories in the root section (starting with 'dir_'),
                    # if defined as relative, are considered relative to `dir_data`
                    if not self.config[k].is_absolute():
                        self.config[k] = self.config["dir_data"] / self.config[k]
                    self.config[k].mkdir(exist_ok=True)

    def get(
        self,
        key: str,
        section: Optional[str] = None,
        *,
        default=None,
        write: bool = False,
        comment: str = "",
    ) -> Any:
        """
        Returns the key if present in core-config.toml
        """

        config_path = self.config_file

        if section is None:
            current_config = self.config
        else:
            if (section not in self.config) and (not write):
                raise KeyError(f"Missing section [{section}] in file {config_path}")
            current_config = self.config[section]

        if key not in current_config:
            if default is None:
                raise KeyError(
                    f"Missing key {key} in section [{section}] of "
                    f"file {config_path} and no defaults provided"
                )
            else:
                if write:
                    assert section is not None
                    self._write_key(config_path, section, key, default, comment)
                return default
                
        value = current_config[key]
        
        # auto convert value to path if its key starts with 'dir_'
        if type(value) is str and key.startswith("dir_"):
            value = Path(value)

        return value 

    @interface
    def _write_key(config_file: Path, section: str, key: str, value, comment: str = ""):

        section = dedent(section.strip())
        key = dedent(key.strip())
        comment = dedent(comment.strip())

        if not comment.startswith("#") and len(comment) > 0:
            comment = f"# {comment} "

        found_section = False
        wait_for_next_line = False

        if type(value) is str:
            value = dedent(value.strip())
            
            if "\n" in value:
                if not value.startswith("\n"):
                    value = "\n" + value
                if not value.endswith("\n"):
                    value = value + "\n"
                value = '""' + value + '""'
            value = f'"{value}"'

        new_config_file = config_file
        config_file = config_file.rename(
            config_file.parent / (str(config_file.name) + ".old")
        )

        # Open the input file for reading and a new file for writing
        with open(config_file, "r") as infile, open(new_config_file, "w+") as outfile:

            # helper functions
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


# single Config instance
try:
    cfg = Config()
except FileNotFoundError:
    # don't fail if config file was not initialized yet
    cfg = None


@click.command()
@click.argument("dir_data")
@click.option("--local", is_flag=True, default=False)
def initialize(dir_data: Path, local: bool):

    mode = "local" if local else "global"
    config_file = get_config_file(mode)

    if config_file.exists():
        error(
            rgb.orange,
            f'{mode} configuration file "{config_file}" is already initialized',
        )
    else:
        # Initialize default config file
        content = dedent(
            f"""
            # HYGEOS general configuration file
            # This file has been initialized by `python -m core.config`

            # Note: all variables starting with 'dir_' are directories. If specified as
            # non-absolute, these directories are assumed relative to `dir_data`
            # => "static" == "<dir_data>/static"

            # Root of the data directory
            # All data in this directory are assumed disposable, and should be
            # downloaded on the fly
            dir_data = "{dir_data}"

            # static data files, required for processing
            dir_static = "static"

            # sample products, used for testing
            dir_samples = "sample_products"

            # ancillary data (downloaded on the fly)
            dir_ancillary = "ancillary"

        """
        )
        with open(config_file, "w") as fp:
            print("Created configuration file", config_file)
            fp.writelines(content)


def get(
    key: str,
    section=None,
    *,
    default=None,
    write: bool = False,
    comment: str = "",
) -> Any:
    """Returns a key from the default module-wide configuration instance

    Args:
        key (str): key name
        section (str, optional): Optional section name. Otherwise the key is taken at
            root level.
        default (optional): Optional default value.
        write (bool, optional): whether to write the default value in the config file.
            Defaults to False.
        comment (str, optional): a comment string to write alongside default value if
            write mode is activated.

    Example:
        from core import config
        config.get('dir_samples')  # returns the sample directory in either local or
                                   # global configuration file
    """
    
    if cfg is None:
        raise configfilenotfound

    return cfg.get(
        key=key, section=section, default=default, write=write, comment=comment
    )


if __name__ == "__main__":
    initialize()
