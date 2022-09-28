from os.path import isfile

import whatthepatch


class Patch:

    def __init__(self, patch_file_path, ignore_empty_lines=True):
        assert isfile(patch_file_path)
        self.patch_file_path = patch_file_path
        self.ignore_empty_lines = ignore_empty_lines
        self.old = None
        self.new = None

    def get_diffs(self):
        with open(self.patch_file_path) as f:
            text = f.read()
            return whatthepatch.parse_patch(text)

    def parse_changed_lines(self, diffs=None, force_rerun=False):
        if diffs is None:
            diffs = self.get_diffs()
        if force_rerun or self.old is None or self.new is None:
            for diff in diffs:
                old_path = diff.header.old_path
                new_path = diff.header.new_path
                changes = diff.changes
                new_numbers = set()
                old_numbers = set()
                previous_change = None
                new_change_fragment_consumed = False
                if changes is None:
                    continue
                for change in changes:
                    if change.old == change.new or self.ignore_empty_lines and (
                            change.line is None or 0 == len(change.line.strip())):
                        new_change_fragment_consumed = False
                        previous_change = change
                        continue
                    if change.new is not None and change.old is None:
                        new_numbers.add(change.new)
                        if previous_change is not None and previous_change.old is not None \
                                and previous_change.new is not None and not new_change_fragment_consumed:
                            old_numbers.add(previous_change.old)
                            new_change_fragment_consumed = True
                    elif change.old is not None and change.new is None:
                        old_numbers.add(change.old)
                        if previous_change is not None and previous_change.old is not None \
                                and previous_change.new is not None and not new_change_fragment_consumed:
                            new_numbers.add(previous_change.new + 1)
                            new_change_fragment_consumed = True
                    else:
                        new_change_fragment_consumed = False
                    previous_change = change


                if self.old is None:
                    self.old = {old_path: old_numbers}
                elif old_path in self.old:
                    self.old.get(old_path).update(old_numbers)
                else:
                    self.old[old_path] = old_numbers

                if self.new is None:
                    self.new = {new_path: new_numbers}
                elif new_path in self.new:
                    self.new.get(new_path).update(new_numbers)
                else:
                    self.new[new_path] = new_numbers

        return self
