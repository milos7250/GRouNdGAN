from pathlib import Path

FAIL_MSG = """
One or more required resource files are missing. Please ensure the following files are present:
    {},
    {},
    {}
    
Alternatively, change the paths in tests/resources/constants.py to point to the correct locations of these
files on your system."""


def test_resources_available():
    from .constants import CAUSAL_GRAPH_FILE, TRAIN_FILE, VALID_FILE

    assert all(Path(file).exists() for file in [CAUSAL_GRAPH_FILE, TRAIN_FILE, VALID_FILE]), FAIL_MSG.format(
        TRAIN_FILE, VALID_FILE, CAUSAL_GRAPH_FILE
    )
