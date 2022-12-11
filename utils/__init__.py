def patch_to_csv(file_path, output_file_path, ignore_empty_lines):
    import os
    abs_path = os.path.abspath(file_path)
    from os.path import isfile
    assert isfile(abs_path), 'could not find file {0}'.format(abs_path)
    from pathlib import Path
    output_path = Path(output_file_path)
    from os.path import isdir
    if not isdir(output_path.parent.absolute()):
        try:
            os.makedirs(output_path.parent.absolute())
        except FileExistsError:
            print("two threads created the directory concurrently.")
    from utils.patch_parser import Patch
    p = Patch(file_path, ignore_empty_lines).parse_changed_lines()
    from dataclasses import make_dataclass
    PatchChangedLines = make_dataclass("PatchChangedLines", [("file", str), ("lines", set()), ("version", int)])
    import pandas as pd
    changes = []
    for file in p.old:
        changes.append(PatchChangedLines(file, p.old[file], 0))
    for file in p.new:
        changes.append(PatchChangedLines(file, p.new[file], 1))
    pd.DataFrame(changes).to_csv(output_path.absolute(), encoding='utf-8', index_label='index')
    return p

