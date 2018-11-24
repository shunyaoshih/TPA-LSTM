import os

import numpy as np
import pypianoroll as ppr
import requests
import scipy.misc
from tqdm import tqdm


def check_path_exists(path):
    return os.path.exists(path)


def create_dir(path):
    try:
        os.makedirs(path)
    except os.error:
        pass


def download_file(url, destination):
    if check_path_exists(destination):
        return
    r = requests.get(url, stream=True)
    with open(destination, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    r.close()


def download_file_from_google_drive(id, destination):
    if check_path_exists(destination):
        return

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
