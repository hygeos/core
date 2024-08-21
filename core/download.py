import subprocess
from pathlib import Path

from core.fileutils import filegen


def download_url(url, dirname, wget_opts='',
                 check_function=None,
                 verbose=True,
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
    @filegen(**kwargs)
    def download_target(path):
        if verbose:
            print('Downloading:', url)
            print('To: ', target)

        cmd = f'wget {wget_opts} {url} -O {path}'
        # Detect access problem
        ret = subprocess.call(cmd.split())
        if ret:
            raise FileNotFoundError(cmd)

        if check_function is not None:
            check_function(path)

    download_target(target)

    return target