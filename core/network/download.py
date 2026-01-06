import subprocess
from pathlib import Path
from typing import Literal

from core.files.fileutils import filegen
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

