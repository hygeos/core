import subprocess
from pathlib import Path
from typing import Literal

from core.files.fileutils import filegen, mdir
from core.files.uncompress import uncompress
from core import log


def download_url(
    url: str,
    dirname: Path,
    wget_opts="",
    check_function=None,
    verbose=True,
    if_exists: Literal['skip', 'overwrite', 'backup', 'error'] = 'skip',
    **kwargs
) -> Path:
    """
    Download `url` to `dirname` with wget

    Options `wget_opts` are added to wget
    Uses a `filegen` wrapper
    Other kwargs are passed to `filegen` (lock_timeout, tmpdir, if_exists)

    Returns the path to the downloaded file
    """
    target = Path(dirname)/(Path(url).name)
    @filegen(if_exists=if_exists, verbose=verbose, **kwargs)
    def download_target(path):
        if verbose:
            log.info('Downloading:', url)
            log.info('To: ', target)

        cmd = f'wget {wget_opts} {url} -O {path}'
        # Detect access problem
        ret = subprocess.call(cmd.split())
        if ret:
            raise FileNotFoundError(cmd)

        if check_function is not None:
            check_function(path)

    download_target(target)

    return target

def download_nextcloud(product_name: str, 
                       output_dir: Path | str, 
                       nextcloud_dir: Path | str = '',
                       sharelink: str = 'https://docs.hygeos.com/s/Fy2bYLpaxGncgPM/', 
                       wget_opts="",
                       check_function=None,
                       verbose: bool = True,
                       if_exists: Literal['skip', 'overwrite', 'backup', 'error'] = 'skip',
                       **kwargs):
    """
    Function for downloading data from Nextcloud contained in the data/eoread directory

    Args:
        product_name (str): Name of the product with the extension
        output_dir (Path | str): Directory where to store downloaded data
        nextcloud_dir (Path | str, optional): Sub Nextcloud repository in which the product are stored. Defaults to ''.
        sharelink (str, optional): Nextcloud public link. By defaults, it is public link to eoread repository.

    Returns:
        Path: Output path of the downloaded data
    """
    
    output_dir = mdir(output_dir)
    inputpath  = Path(nextcloud_dir)/product_name
    
    url = f'{sharelink}/download?files={inputpath}'
    target = output_dir/product_name
    @filegen(if_exists=if_exists, verbose=verbose, **kwargs)
    def download_target(path):
        if verbose:
            log.info('Downloading:', url)
            log.info('To: ', target)

        cmd = f'wget {wget_opts} {url} -O {path}'
        # Detect access problem
        ret = subprocess.call(cmd.split())
        if ret:
            raise FileNotFoundError(cmd)

        if check_function is not None:
            check_function(path)

    download_target(target)
    
    # Uncompress downloaded file 
    if product_name.split('.')[-1] in ['zip','gz']:
        return uncompress(target, output_dir)
        
    return target