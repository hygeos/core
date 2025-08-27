#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gzip
from datetime import timedelta
from pathlib import Path
import pytest
from tempfile import NamedTemporaryFile, TemporaryDirectory
import zipfile
from core.files.uncompress import CacheDir, duration, uncompress


def test_uncompress_cache():
    with TemporaryDirectory(prefix='test_uncompress_cache') as tmpdir, \
            NamedTemporaryFile(suffix='.gz') as tmpfile:

        # create some compressed file
        with gzip.open(tmpfile, 'w') as fp:
            fp.write(b'Sample file')
        
        # uncompress a file twice
        # the returned path should be identical
        cdir = CacheDir(tmpdir)
        path1 = cdir.uncompress(tmpfile.name)
        path2 = cdir.uncompress(tmpfile.name)
        assert path1 == path2
        assert path1.exists()


def test_uncompress_uncompressed():
    # passing an uncompressed file should return the same file
    with TemporaryDirectory(prefix='test_uncompress_cache') as tmpdir, \
            NamedTemporaryFile() as tmpfile:
        # write a sample file
        with open(tmpfile.name, 'w') as fp:
            fp.write('Sample file')

        cdir = CacheDir(tmpdir)
        assert cdir.uncompress(tmpfile.name) == Path(tmpfile.name)
        

def test_duration():
    assert duration('2w') == timedelta(weeks=2)
    assert duration('2d') == timedelta(days=2)
    assert duration('2h') == timedelta(hours=2)

@pytest.mark.parametrize('nb_file', [1,2])
def test_uncompress_files(nb_file):
    """
    Test that uncompress works on test.zip with (0, 1, or 2 files)
    """
    def create_zipfile_with_files(path, nb_file, zip_name):
        for i in range(nb_file):
            with open(path / ("t" + str(i) + ".txt"), 'w') as fd:
                fd.write('test')
            
        with zipfile.ZipFile(path / zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i in range(nb_file):
                zipf.write(path / ("t" + str(i) + ".txt"), arcname=("t" + str(i) + ".txt"))
        
        return path / zip_name

    with TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)
        zip_name = "text.zip"
        zip_path = create_zipfile_with_files(target, nb_file, zip_name)
        
        result_uncomp = Path(uncompress(zip_path, target/"result", create_out_dir=True, verbose=True))
        
        print("res", result_uncomp)
        
        assert result_uncomp.exists()
        
        path = Path(tmpdir)
        for item in path.rglob("*"):
            print(item)
            
        assert sum(1 for f in result_uncomp.iterdir() if f.is_file()) == nb_file
        
@pytest.mark.parametrize('nb_folder', [1,2])
def test_uncompress_folders(nb_folder):
    """
    Test that uncompress works on test.zip with (0, 1, or 2 files)
    """
    def create_zipfile_with_files(path, nb_folder, zip_name):
        path = Path(path)
    
        # Create folders and files
        for i in range(nb_folder):
            folder = path / f"folder{i}"
            folder.mkdir(parents=True, exist_ok=True)
            file_path = folder / f"t{i}.txt"
            with file_path.open('w') as fd:
                fd.write('test')
        
        # Create the zip archive
        zip_path = path / zip_name
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i in range(nb_folder):
                file = path / f"folder{i}" / f"t{i}.txt"
                # Preserve relative folder structure
                zipf.write(file, arcname=file.relative_to(path))
        
        return zip_path

    with TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)
        zip_name = "folder.zip"
        zip_path = create_zipfile_with_files(target, nb_folder, zip_name)
        
        result_uncomp = Path(uncompress(zip_path, target/"result", create_out_dir=True, verbose=True))
        
        print("res", result_uncomp)
        
        assert result_uncomp.exists()
        
        path = Path(tmpdir)
        for item in path.rglob("*"):
            print(item)
        
        print(sum(1 for f in result_uncomp.iterdir() if f.is_dir()))
        
        assert sum(1 for f in result_uncomp.iterdir() if f.is_dir()) == nb_folder


def test_uncompress_same_name_folder():
    """
    Test that uncompress works on test.zip with (0, 1, or 2 files)
    """
    def create_zipfile_with_files(path, zip_name):
        path = Path(path)
    
        # Create folders and files
        folder = path / "folder"
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / "t.txt"
        with file_path.open('w') as fd:
            fd.write('test')
        
        # Create the zip archive
        zip_path = path / zip_name
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            file = path / "folder" / "t.txt"
            # Preserve relative folder structure
            zipf.write(file, arcname=file.relative_to(path))
        
        return zip_path

    with TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)
        zip_name = "folder.zip"
        zip_path = create_zipfile_with_files(target, zip_name)
        
        result_uncomp = Path(uncompress(zip_path, target/"folder", create_out_dir=True, verbose=True))
        
        print("res", result_uncomp)
        
        assert result_uncomp.exists()
        
        list_files_in_result = list(result_uncomp.rglob("*"))
        
        assert len(list_files_in_result) == 1
