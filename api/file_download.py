from pathlib import Path

import gdown

# CONSTANT VARIABLE
UTIL_FOLDER_NAME = "utils"
FOLDER_ID = "1zSbS2qicwE5oirylQjE2Qm_AG6-i27Rc"


def download_file(file_id, destination):
    """Automatically download checkpoints from google drive"""
    if not Path(Path(".") / UTIL_FOLDER_NAME).exists():
        Path(UTIL_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}",
        destination,
        quiet=False,
        fuzzy=True,
    )


if __name__ == "__main__":
    download_file()
