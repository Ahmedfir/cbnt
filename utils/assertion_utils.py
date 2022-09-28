def assert_not_empty(*args):
    for arg in args:
        assert not is_empty(arg), 'Passed empty or None argument!'


def is_empty(arg):
    return arg is None or len(arg) == 0


def is_empty_strip(arg: str):
    return arg is None or len(arg.strip()) == 0
