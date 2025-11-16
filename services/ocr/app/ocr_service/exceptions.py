class OCRException(Exception):
    """Base OCR exception."""


class FileNotProvidedError(OCRException):
    pass


class UnsupportedFileTypeError(OCRException):
    pass


class OCRProcessException(OCRException):
    pass


class StorageUnavailableError(OCRException):
    pass
