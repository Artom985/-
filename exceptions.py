import os

class NeuralNetworkError(Exception):
    """Базовый класс для всех исключений нейронной сети."""
    pass

class InvalidLayerSizeError(NeuralNetworkError):
    """Возникает при некорректном указании размеров слоев."""
    pass

class MismatchedDataError(NeuralNetworkError):
    """Возникает при несовпадении размерностей входных данных и весов."""
    pass

class ActivationNotFoundError(NeuralNetworkError):
    """Возникает при попытке использовать несуществующую функцию активации."""
    pass

class DatasetError(Exception):
    """Базовый класс для исключений менеджера данных."""
    pass

class FileNotFoundCustomError(DatasetError):
    """Возникает, если указанный CSV-файл не найден."""
    def __init__(self, path):
        super().__init__(f"Файл не найден: {path}")
        self.path = path

class InvalidCSVFormatError(DatasetError):
    """Возникает при ошибках парсинга CSV или пустом файле."""
    pass

class InvalidSplitRatioError(DatasetError):
    """Возникает при некорректной пропорции разделения данных."""
    pass