from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


def read_image(file: UploadFile) -> Image.Image:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Type de fichier non supporté ({file.content_type}). Formats acceptés : JPEG, PNG.",
        )
    try:
        image = Image.open(file.file)
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Image corrompue ou illisible.")
    return image
