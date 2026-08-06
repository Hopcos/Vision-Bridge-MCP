"""validators 与错误模型单元测试。"""

from __future__ import annotations

import pytest

from vision_bridge.errors import (
    Base64DecodeError,
    ImageFormatError,
    ImageSizeError,
    ImageSourceError,
    URLBlockedError,
    VisionError,
)
from vision_bridge import validators as v
from vision_bridge.validators import (
    is_supported_magic,
    is_supported_extension,
    is_supported_mime,
    sniff_image_format,
    validate_base64,
    validate_file_size,
    validate_http_url,
    is_private_host,
)


class TestMagicBytes:
    def test_png(self):
        png = b"\x89PNG\r\n\x1a\n" + b"x" * 16
        assert sniff_image_format(png) == "PNG"
        assert is_supported_magic(png)

    def test_jpeg(self):
        jpg = b"\xff\xd8\xff\xe0" + b"x" * 16
        assert sniff_image_format(jpg) == "JPEG"

    def test_unknown_rejected(self):
        assert not is_supported_magic(b"<html> not an image</html>")

    def test_gif_bmp_webp(self):
        assert sniff_image_format(b"GIF89a" + b"x" * 16) == "GIF"
        assert sniff_image_format(b"BM" + b"x" * 16) == "BMP"
        assert sniff_image_format(b"RIFF" + b"x" * 8 + b"WEBP") == "WEBP"


class TestExtensionsAndMime:
    def test_supported_ext(self):
        assert is_supported_extension("a.png")
        assert is_supported_extension("b.JPG")
        assert not is_supported_extension("c.zip")

    def test_supported_mime(self):
        assert is_supported_mime("image/png")
        assert is_supported_mime("image/jpeg")
        assert not is_supported_mime("text/html")


class TestSizeValidation:
    def test_too_big(self):
        with pytest.raises(ImageSizeError):
            validate_file_size(1024 * 1024 * 21, max_size=20 * 1024 * 1024)

    def test_ok(self):
        validate_file_size(1000, max_size=20 * 1024 * 1024)

    def test_zero_rejected(self):
        with pytest.raises(ImageSizeError):
            validate_file_size(0, max_size=20 * 1024 * 1024)


class TestBase64Validation:
    def test_data_url(self):
        val = validate_base64("data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==")
        assert val == "iVBORw0KGgoAAAANSUhEUg=="

    def test_plain_base64(self):
        assert validate_base64("aGVsbG8=") == "aGVsbG8="

    def test_invalid(self):
        assert validate_base64("!!!not base64!!!") is None
        assert validate_base64("") is None

    def test_wrong_mime_rejected(self):
        assert validate_base64("data:application/x-msdownload;base64,aGVsbG8=") is None


class TestHTTPUrlSecurity:
    def test_private_loopback_blocked(self):
        with pytest.raises(URLBlockedError):
            validate_http_url("http://127.0.0.1:8001/v1/models")

    def test_domain_localhost_blocked(self):
        with pytest.raises(URLBlockedError):
            validate_http_url("http://localhost/image.png")

    def test_private_ip_blocked(self):
        with pytest.raises(URLBlockedError):
            validate_http_url("http://192.168.1.5/image.png")

    def test_public_allowed(self):
        url = "https://example.com/image.png"
        assert validate_http_url(url) == url

    def test_bad_scheme(self):
        with pytest.raises(ImageSourceError):
            validate_http_url("file:///etc/passwd")

    def test_bad_host(self):
        with pytest.raises(ImageSourceError):
            validate_http_url("http:///nohost")


class TestErrors:
    def test_hierarchy(self):
        assert issubclass(ImageFormatError, VisionError)
        assert issubclass(ImageSourceError, VisionError)
        assert issubclass(Base64DecodeError, VisionError)

    def test_code_attr(self):
        assert ImageFormatError("x", ).code == "ImageFormat"
