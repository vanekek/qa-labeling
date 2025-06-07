import os
from pathlib import Path

import requests


def download_from_github(repo_url, file_names, dest_folder="data"):
    """
    Загружаем файлы из GitHub репозитория

    :param repo_url: URL репозитория (например 'https://github.com/username/reponame')
    :param file_names: список имен файлов для загрузки
    :param dest_folder: локальная папка для сохранения
    """

    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    if "github.com" in repo_url:
        raw_base = (
            repo_url.replace("github.com", "raw.githubusercontent.com") + "/main/"
        )
    else:
        raw_base = repo_url + "/"

    for file_name in file_names:
        file_url = raw_base + file_name
        file_path = os.path.join(dest_folder, file_name)

        try:
            response = requests.get(file_url)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"✓ Файл {file_name} успешно загружен в {dest_folder}/")
        except requests.exceptions.RequestException as e:
            print(f"✕ Ошибка при загрузке {file_name}: {e}")


def main():
    repo_url = "https://github.com/vanekek/q-a-labeling-data"
    files_to_download = ["train.csv", "val.csv", "test.csv"]
    destination_folder = "data_raw"

    download_from_github(repo_url, files_to_download, destination_folder)


if __name__ == "__main__":
    main()
