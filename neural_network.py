import numpy as np
import json
import os

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='sigmoid', seed=None):
        """
        :param layer_sizes: Список целых чисел, например [2, 5, 1] (2 входа, 5 нейронов в скрытом слое, 1 выход).
        :param activation: 'sigmoid' или 'relu'.
        :param seed: Сид для воспроизводимости.
        """
        if len(layer_sizes) < 3:
            raise InvalidLayerSizeError("Сеть должна иметь минимум входной, один скрытый и выходной слои.")
        if not all(isinstance(x, int) and x > 0 for x in layer_sizes):
            raise InvalidLayerSizeError("Размеры слоев должны быть положительными целыми числами.")
        
        self.layer_sizes = layer_sizes
        self.activation_name = activation.lower()
        self._init_activation()
        
        self.weights = []
        self.biases = []
        self.history = {'loss': []}
        
        rng = np.random.default_rng(seed)
        for i in range(len(layer_sizes) - 1):
            # Инициализация по методу Ксавье/Глорота для сигмоиды
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            self.weights.append(rng.uniform(-limit, limit, (layer_sizes[i+1], layer_sizes[i])))
            self.biases.append(np.zeros((layer_sizes[i+1], 1)))

    def _init_activation(self):
        if self.activation_name == 'sigmoid':
            self.activation = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative
        elif self.activation_name == 'relu':
            self.activation = self._relu
            self.activation_derivative = self._relu_derivative
        else:
            raise ActivationNotFoundError(f"Функция активации '{self.activation_name}' не поддерживается.")

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _sigmoid_derivative(x):
        return x * (1 - x)  # x здесь уже после сигмоиды

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _relu_derivative(x):
        return (x > 0).astype(float)

    def forward(self, X):
        """Прямое распространение. X должен быть в форме (n_features, n_samples)."""
        if X.shape[0] != self.layer_sizes[0]:
            raise MismatchedDataError(f"Ожидалось {self.layer_sizes[0]} признаков, получено {X.shape[0]}.")
            
        self.activations = [X]
        self.z_values = []
        
        A = X
        for i in range(len(self.weights) - 1):
            Z = self.weights[i] @ A + self.biases[i]
            self.z_values.append(Z)
            A = self.activation(Z)
            self.activations.append(A)
        
        # Выходной слой (линейная активация для регрессии или сигмоида для бинарной классификации)
        Z_out = self.weights[-1] @ A + self.biases[-1]
        self.z_values.append(Z_out)
        A_out = Z_out  # Для регрессии. Если нужна классификация, можно вернуть self._sigmoid(Z_out)
        self.activations.append(A_out)
        
        return A_out

    def backward(self, X, Y, learning_rate):
        """Обратное распространение. Y в форме (n_outputs, n_samples)."""
        m = X.shape[1]
        
        # Производная функции потерь MSE: dL/dA_out = (A_out - Y) / m
        dA = (self.activations[-1] - Y) / m
        
        # Градиенты для выходного слоя
        dZ = dA  # Так как активация на выходе линейная
        dW = dZ @ self.activations[-2].T
        db = np.sum(dZ, axis=1, keepdims=True)
        
        self.weights[-1] -= learning_rate * dW
        self.biases[-1] -= learning_rate * db
        
        # Градиенты для скрытых слоев (двигаемся в обратном порядке)
        dA_prev = self.weights[-1].T @ dZ
        for i in range(len(self.weights) - 2, -1, -1):
            dZ = dA_prev * self.activation_derivative(self.activations[i+1])
            dW = dZ @ self.activations[i].T
            db = np.sum(dZ, axis=1, keepdims=True)
            
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            if i != 0:
                dA_prev = self.weights[i].T @ dZ

    def train(self, X, Y, epochs, learning_rate, batch_size=None):
        """Обучение сети."""
        if batch_size is None or batch_size > X.shape[1]:
            batch_size = X.shape[1]

        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[1])
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            for i in range(0, X.shape[1], batch_size):
                end = min(i + batch_size, X.shape[1])
                X_batch = X_shuffled[:, i:end]
                Y_batch = Y_shuffled[:, i:end]
                
                self.forward(X_batch)
                self.backward(X_batch, Y_batch, learning_rate)

            # Сохранение истории ошибок (MSE)
            predictions = self.forward(X)
            loss = np.mean((predictions - Y) ** 2)
            self.history['loss'].append(loss)

    def predict(self, X):
        return self.forward(X)

    def save_weights(self, filepath):
        """Сохранение весов и смещений в JSON."""
        data = {
            'layer_sizes': self.layer_sizes,
            'activation': self.activation_name,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def load_weights(self, filepath):
        """Загрузка весов из JSON."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл весов не найден: {filepath}")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if data['layer_sizes'] != self.layer_sizes or data['activation'] != self.activation_name:
            raise MismatchedDataError("Архитектура сети в файле не совпадает с текущей.")
            
        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]