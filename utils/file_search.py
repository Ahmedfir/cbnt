def contains(file_path, query, encoding='utf-8') -> bool:
    import mmap
    # @see https://stackoverflow.com/a/4944929/3014036
    with open(file_path, 'rb', 0) as file, \
            mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
        if isinstance(query, list):
            if all([s.find(bytes(q, encoding=encoding)) != -1 for q in query]):
                return True
        elif s.find(bytes(query, encoding=encoding)) != -1:
            return True
    return False

