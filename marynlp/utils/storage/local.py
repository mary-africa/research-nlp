import os
import tempfile
from pathlib import Path
from typing import Union

import logging
logger = logging.getLogger('marynlp')


__ROOT_STORAGE_PATH_NAME = './.marynlp/store'

# Root storage path for storing the models
__ROOT_STORAGE_PATH: Path = str(Path.home().joinpath(__ROOT_STORAGE_PATH_NAME))

def get_path_from_store(relative_path: str):
    """Get the path of the file as it would have been from the `marynlp` store"""
    pt = Path('./').joinpath(relative_path)
    return Path(__ROOT_STORAGE_PATH).joinpath(pt)

def get_temp_path(path: Union[str, os.PathLike]):
    """Get the temporary path."""
    temp_path = Path(tempfile.TemporaryDirectory().name)
    temp_path = temp_path.joinpath(f'./{path}')
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    return str(temp_path)
