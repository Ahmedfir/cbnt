from os import makedirs
from typing import List


def write_file(file_path: str, content: str, encoding="utf-8"):
    with open(file_path, 'w', encoding=encoding) as file:
        return file.write(content)


def load_file(file_path: str, encoding="utf-8") -> str:
    with open(file_path, encoding=encoding) as file:
        return file.read()


def load_file_lines(file_path: str) -> List[str]:
    with open(file_path) as f:
        return f.readlines()


def write_csv_row(file_path, row):
    import csv
    with open(file_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(row)


def copy_dir(src, dest, create_dir=True, remove_src_after=False):
    import shutil
    from os.path import join, isdir
    from pathlib import Path
    assert isdir(src), 'src not dir : ' + str(src)
    dest_dir = join(dest, Path(src).name) if create_dir else dest
    shutil.copytree(src, dest_dir)
    if remove_src_after:
        shutil.rmtree(src)

