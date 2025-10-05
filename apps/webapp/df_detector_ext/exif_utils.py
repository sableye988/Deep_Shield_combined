def is_photoshop_like(img):
    exif = img.getexif()
    if not exif:
        return False
    for tag, value in exif.items():
        if isinstance(value, str) and any(k in value.lower() for k in ["photoshop","adobe","lightroom"]):
            return True
    return False
