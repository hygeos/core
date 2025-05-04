import subprocess
from os import system
from pathlib import Path
from typing import Literal

from core.fileutils import filegen, mdir
from core.uncompress import uncompress
from core import log

sharelink_eoread = 'https://docs.hygeos.com/s/Fy2bYLpaxGncgPM/'

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
                       input_dir: Path | str = '',
                       verbose: bool = True,
                       if_exists: Literal['skip', 'overwrite', 'backup', 'error'] = 'skip'):
    """
    Function for downloading data from Nextcloud contained in the data/eoread directory

    Args:
        product_name (str): Name of the product with the extension
        output_dir (Path | str): Directory where to store downloaded data
        input_dir (Path | str, optional): Sub repository in which the product are stored. Defaults to ''.

    Returns:
        Path: Output path of the downloaded data
    """
    
    output_dir = mdir(output_dir)
    outpath    = output_dir/product_name
    inputpath  = Path(input_dir)/product_name
    
    url = f'{sharelink_eoread}/download?files={inputpath}'
    download_url(url, outpath, wget_opts='-c', verbose=verbose, if_exists=if_exists)
    
    # Uncompress downloaded file 
    if product_name.split('.')[-1] in ['zip','gz']:
        return uncompress(outpath, output_dir)
        
    return outpath