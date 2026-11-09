import shutil
from pathlib import Path

from fastapi import UploadFile

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploaded_docs"


def save_uploaded_files(files: list[UploadFile]) -> list[Path]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_paths = []
    for file in files:
        filename = Path((file.filename or "upload.pdf").replace("\\", "/")).name
        file_path = UPLOAD_DIR / filename
        with file_path.open("wb") as saved_file:
            shutil.copyfileobj(file.file, saved_file)
        file_paths.append(file_path)
    return file_paths