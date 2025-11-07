import os

def what(file):
    """Simple replacement for imghdr.what() — detects JPEG/PNG/GIF."""
    if isinstance(file, str):
        if not os.path.isfile(file):
            return None
        with open(file, "rb") as f:
            head = f.read(10)
    else:
        head = file.read(10)
        file.seek(0)

    if head.startswith(b'\xff\xd8'):
        return 'jpeg'
    elif head.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    elif head[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    else:
        return None
