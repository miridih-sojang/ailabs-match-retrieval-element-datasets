import os

import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO

import urllib.request
from urllib.parse import quote

from utils import get_args, read_yaml
tqdm.pandas()


def encode_url(url):
    parts = url.split('/')
    last_part_encoded = quote(parts[-1])

    parts[-1] = last_part_encoded
    new_url = '/'.join(parts)

    return new_url


def download_image(url, folder_path, file_path, save_path):
    response = urllib.request.urlopen(f'{url}/{file_path}')
    image_data = response.read()
    image = Image.open(BytesIO(image_data))
    png_file_path = os.path.join(save_path, folder_path, os.path.basename(file_path) + '.png')
    image.save(png_file_path, 'PNG')
    return png_file_path


def get_download_image_path(folder_path, file_path, save_path):
    png_file_path = os.path.join(save_path, folder_path, os.path.basename(file_path) + '.png')
    return png_file_path


def main():
    args = get_args()
    config = read_yaml(args.config_path)
    FILE_NAME, AWS_URL, SAVE_PATH, CASE_PATH = config['FILE_NAME'], config['AWS_URL'], config['SAVE_PATH'], config[
        'CASE_PATH']
    df = pd.read_csv(FILE_NAME)
    df.dropna(inplace=True)
    df['folder_path'] = df['file_path'].progress_apply(lambda x: '/'.join(x.split('/')[:-1]))
    print(df.shape)
    for file_path, folder_path in tqdm(df[['file_path', 'folder_path']].values):
        file_save_path = file_path.replace("/", "_")
        png_file_path = get_download_image_path(folder_path, file_path, SAVE_PATH)
        if os.path.exists(png_file_path):
            continue
        try:
            os.makedirs(f'{SAVE_PATH}/{folder_path}', exist_ok=True)
            png_file_path = download_image(AWS_URL, folder_path, file_path, SAVE_PATH)
            f = open(f'{CASE_PATH}/success_case.txt', 'a')
            f.write(f'{file_path}\n')
            f.write(f'{file_path}\n')
            f.write(f'{file_save_path}\n')
            f.write(f'{png_file_path}\n')
            f.close()
        except Exception as e_1:
            try:
                new_file_path = encode_url(file_path)
                _ = download_image(AWS_URL, folder_path, new_file_path, SAVE_PATH)
                f = open(f'{CASE_PATH}/success_case.txt', 'a')
                f.write(f'{file_path}\n')
                f.write(f'{new_file_path}\n')
                f.write(f'{file_save_path}\n')
                f.write(f'{png_file_path}\n')
                f.close()
            except Exception as e_2:
                f = open(f'{CASE_PATH}/fail_case.txt', 'a')
                f.write(f'{file_path}\n')
                f.write(f'{file_save_path}\n')
                f.write(f'Error 1 : {e_1}\n')
                f.write(f'Error 2 : {e_2}\n\n\n')
                f.close()


if __name__ == '__main__':
    main()
