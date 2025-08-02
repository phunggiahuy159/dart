import os
import zipfile
from torchvision.datasets.utils import download_url
import logging

class Logger:
    def __init__(self, level):
        self.logger = logging.getLogger('TopMost')
        self.set_level(level)
        self._add_handler()
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(f"{message}")

    def warning(self, message):
        self.logger.warning(f"WARNING: {message}")

    def set_level(self, level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in levels:
            self.logger.setLevel(level)

    def _add_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(sh)

        # Remove duplicate handlers
        if len(self.logger.handlers) > 1:
            self.logger.handlers = [self.logger.handlers[0]]


logger = Logger("WARNING")


def download_dataset(dataset_name, cache_path="~/.topmost"):
    """
    Download dataset from TopMost repository
    
    Args:
        dataset_name (str): Name of the dataset to download
        cache_path (str): Path to cache the downloaded dataset
    """
    cache_path = os.path.expanduser(cache_path)
    raw_filename = f'{dataset_name}.zip'

    if dataset_name in ['Wikitext-103']:
        # download from Git LFS.
        zipped_dataset_url = f"https://media.githubusercontent.com/media/BobXWu/TopMost/main/data/{raw_filename}"
    else:
        zipped_dataset_url = f"https://raw.githubusercontent.com/BobXWu/TopMost/master/data/{raw_filename}"

    logger.info(zipped_dataset_url)

    download_url(zipped_dataset_url, root=cache_path, filename=raw_filename, md5=None)

    path = f'{cache_path}/{raw_filename}'
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(cache_path)

    os.remove(path)