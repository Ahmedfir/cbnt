# Custom Decorator function
# @see https://stackoverflow.com/a/60980685/3014036
def list_to_tuple(function):
    def wrapper(*args):
        args = [tuple(x) if type(x) == list else x for x in args]
        result = function(*args)
        result = tuple(result) if type(result) == list else result
        return result

    return wrapper


def tuple_to_list(function):
    def wrapper(*args):
        args = [[*x] if type(x) == tuple else x for x in args]
        result = function(*args)
        result = [*result] if type(result) == tuple else result
        return result

    return wrapper
