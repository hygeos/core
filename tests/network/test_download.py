from pathlib import Path
from tempfile import TemporaryDirectory
from core.network.download import download_url


def test_download_url():
    """
    Test the download of a sample file
    """
    with TemporaryDirectory() as tmpdir:
        path = download_url('https://httpbin.org/image/png', Path(tmpdir))
        assert path.exists()
        assert path.name == 'png'