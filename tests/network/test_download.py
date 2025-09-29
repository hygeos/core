from tempfile import TemporaryDirectory
from core.network.download import download_nextcloud


def test_download_nextcloud():
    with TemporaryDirectory() as tmpdir:
        
        # Check downloading
        path = download_nextcloud('test.txt', tmpdir)
        assert path.exists(), 'File does not exists'
        
        # Check content
        with open(path,'r') as f:
            lines = f.readlines()
            assert len(lines) > 0, 'No content found'