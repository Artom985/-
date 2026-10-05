import numpy as np
import pandas as pd
import os

class DatasetManager:
    def __init__(self):
        self.X = None
        self.y = None
        self.feature_mean = None
        self.feature_std = None

    def load_from_csv(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundCustomError(filepath)
        
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                raise InvalidCSVFormatError("CSV-файл пуст.")
        except Exception as e:
            raise InvalidCSVFormatError(f"Ошибка чтения CSV: {e}")

        if df.shape[1] < 2:
            raise InvalidCSVFormatError("В файле должно быть минимум два столбца (признаки и цель).")
            
        self.y = df.iloc[:, -1].values.reshape(-1, 1)
        self.X = df.iloc[:, :-1].values
        
        # Если целевая переменная бинарная (0 или 1), оставляем как есть. 
        # Если регрессия с широким диапазоном, можно добавить нормализацию и для y.
        return self.X, self.y

    def normalize_features(self, method='standard'):
        if self.X is None:
            raise DatasetError("Данные не загружены.")
            
        if method == 'standard':
            self.feature_mean = np.mean(self.X, axis=0)
            self.feature_std = np.std(self.X, axis=0)
            # Защита от деления на ноль
            self.feature_std[self.feature_std == 0] = 1.0
            self.X = (self.X - self.feature_mean) / self.feature_std
        elif method == 'minmax':
            min_val = np.min(self.X, axis=0)
            max_val = np.max(self.X, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            self.X = (self.X - min_val) / range_val
        else:
            raise ValueError("Метод нормализации должен быть 'standard' или 'minmax'.")

    def train_test_split(self, test_size=0.2, shuffle=True, random_state=None):
        if self.X is None or self.y is None:
            raise DatasetError("Данные не загружены.")
        if not 0 < test_size < 1:
            raise InvalidSplitRatioError("test_size должен быть в диапазоне (0, 1).")
            
        m = self.X.shape[0]
        indices = np.arange(m)
        
        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(indices)
            
        split_idx = int(m * (1 - test_size))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        X_train, X_test = self.X[train_idx], self.X[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]
        
        # Нейросеть ожидает форму (n_features, n_samples)
        return X_train.T, X_test.T, y_train.T, y_test.T