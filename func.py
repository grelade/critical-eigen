import os
import sys

def determine_unique_postfix(fn) -> str:
    """
    Determine the unique postfix for a file or directory in order to avoid overwriting
    directories created during the run.
    """
    if not os.path.exists(fn):
        return ""
    path, name = os.path.split(fn)
    name, ext = os.path.splitext(name)
    make_fn = lambda i: os.path.join(path, "{}_{}{}".format(name, i, ext))
    for i in range(1, sys.maxsize):
        uni_fn = make_fn(i)
        if not os.path.exists(uni_fn):
            return "_" + str(i)