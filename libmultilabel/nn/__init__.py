try:
    import torch
except ImportError:
    import sys

    print(
        "Packages under libmultilabel.nn requires additional dependencies.\n"
        "Please install them with\n"
        "    pip install libmultilabel[nn]\n",
        file=sys.stderr,
    )
    raise
