"""
Lightweight imghdr shim for environments missing the stdlib imghdr.
Provides a minimal `what(file, h=None)` implementation covering common types.
This file is intentionally simple and intended for deployment compatibility only.
"""
from typing import Optional

try:
    # Prefer PIL if available for robust detection
    from PIL import Image
except Exception:
    Image = None

def _match_header(h: bytes) -> Optional[str]:
    if not h or len(h) < 4:
        return None
    # JPEG
    if h.startswith(b"\xff\xd8\xff"):
        return 'jpeg'
    # PNG
    if h.startswith(b"\x89PNG\r\n\x1a\n"):
        return 'png'
    # GIF
    if h[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    # BMP
    if h.startswith(b'BM'):
        return 'bmp'
    # TIFF (little/big endian)
    if h.startswith(b'II') and h[2:4] == b'\x2a\x00':
        return 'tiff'
    if h.startswith(b'MM') and h[2:4] == b'\x00\x2a':
        return 'tiff'
    # WebP (RIFF....WEBP)
    if h.startswith(b'RIFF') and h[8:12] == b'WEBP':
        return 'webp'
    return None


def what(file, h: bytes = None) -> Optional[str]:
    """Determine the type of an image.

    Parameters
    - file: filename (str) or a file-like object supporting .read().
    - h: optional initial bytes to inspect.

    Returns the image type string (e.g., 'jpeg', 'png') or None.
    """
    # If PIL is available and file is a filename, use it first
    if Image is not None and isinstance(file, str):
        try:
            with Image.open(file) as im:
                fmt = im.format
                if fmt:
                    return fmt.lower()
        except Exception:
            pass

    data = None
    # If h bytes provided, try header matching first
    if h:
        try:
            data = bytes(h)
        except Exception:
            data = None
        if data:
            typ = _match_header(data)
            if typ:
                return typ
    # If file is filename, read initial bytes
    try:
        if isinstance(file, str):
            with open(file, 'rb') as f:
                head = f.read(32)
                typ = _match_header(head)
                if typ:
                    return typ
        else:
            # file-like object
            pos = None
            try:
                pos = file.tell()
            except Exception:
                pos = None
            head = file.read(32)
            try:
                if pos is not None:
                    file.seek(pos)
            except Exception:
                pass
            typ = _match_header(head)
            if typ:
                return typ
    except Exception:
        pass

    # fallback: try PIL on file-like
    if Image is not None:
        try:
            # If file-like, PIL can accept file-like objects
            if not isinstance(file, str):
                file.seek(0)
                with Image.open(file) as im:
                    fmt = im.format
                    if fmt:
                        return fmt.lower()
        except Exception:
            pass

    return None
