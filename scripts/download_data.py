import os

import requests


def download_folder(repo_url, folder_path, output_dir="data"):
    """
    Загружаем файлы из GitHub репозитория
    """

    api_url = (
        repo_url.replace("https://github.com", "https://api.github.com/repos")
        + "/contents/"
        + folder_path
    )

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        contents = response.json()

        os.makedirs(output_dir, exist_ok=True)

        for item in contents:
            if item["type"] == "file":
                file_url = item["download_url"]
                file_path = os.path.join(output_dir, item["name"])

                file_response = requests.get(file_url)
                file_response.raise_for_status()

                with open(file_path, "wb") as f:
                    f.write(file_response.content)

            elif item["type"] == "dir":
                new_folder_path = os.path.join(folder_path, item["name"])
                new_output_dir = os.path.join(output_dir, item["name"])
                download_folder(repo_url, new_folder_path, new_output_dir)

        print(f"\nВсе файлы успешно скачаны в папку: {os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")


def main():
    repo_url = "https://github.com/vanekek/q-a-labeling-data"
    files_to_download = "dvcstore"
    destination_folder = "tmp/dvcstore/"

    download_folder(repo_url, files_to_download, destination_folder)


if __name__ == "__main__":
    main()
