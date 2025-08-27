from tempfile import TemporaryDirectory
from core.network.download import download_nextcloud


def test_download_nextcloud():
    with TemporaryDirectory() as tmpdir:
        path = download_nextcloud('test.txt', tmpdir)
        assert path.exists()