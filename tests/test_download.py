from tempfile import TemporaryDirectory
from core.download import *


def test_download_nextcloud():
    with TemporaryDirectory() as tmpdir:
        path = download_nextcloud('test.txt', tmpdir)
        assert path.exists()