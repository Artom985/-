import turtle

import numpy as np 
import matplotlib.pyplot as plt # type: ignore
from neural_network import NeuralNetwork
from dataset_manager import DatasetManager
from exceptions import *

def get_float(prompt, default=None):
    while True:
        try:
            val = input(prompt)
            if not val and default is not None:
                return default
            return float(val)
        except ValueError:
            print("Введите корректное число.")

def get_int(prompt, default=None):
    while True:
        try:
            val = input(prompt)
            if not val and default is not None:
                return default
            return int(val)
        except ValueError:
            print("Введите корректное целое число.")

def get_list_of_ints(prompt):
    while True:
        try:
            s = input(prompt)
            return [int(x.strip()) for x in s.split(',')]
        except ValueError:
            print("Введите числа через запятую, например: 2, 10, 1")

def main():
    nn = None
    dm = DatasetManager()
    X_train = X_test = y_train = y_test = None

    while True:
        print("\n=== Меню нейронной сети ===")
        print("1. Создать сеть")
        print("2. Загрузить данные из CSV")
        print("3. Обучить сеть")
        print("4. Сделать предсказание")
        print("5. Сохранить веса")
        print("6. Загрузить веса")
        print("7. Показать график ошибки")
        print("8. Выход")
        
        choice = input("Выберите действие: ")

        if choice == '1':
            try:
                sizes = get_list_of_ints("Введите размеры слоев через запятую (например: 2, 16, 8, 1): ")
                activation = input("Функция активации (sigmoid/relu) [sigmoid]: ") or 'sigmoid'
                seed = get_int("Сид для случайности (оставьте пустым для случайного): ", None)
                if seed is not None and seed == 0: # Если ввели 0
                    seed = 0
                elif seed is None:
                    seed = np.random.randint(0, 10000)
                    
                nn = NeuralNetwork(sizes, activation, seed)
                print(f"Сеть создана. Архитектура: {sizes}, активация: {activation}")
            except (InvalidLayerSizeError, ActivationNotFoundError) as e:
                print(f"Ошибка: {e}")

        elif choice == '2':
            try:
                path = input("Введите путь к CSV-файлу: ")
                dm.load_from_csv(path)
                norm_method = input("Метод нормализации (standard/minmax) [standard]: ") or 'standard'
                dm.normalize_features(norm_method)
                
                ratio = get_float("Доля тестовой выборки (0.0 - 1.0) [0.2]: ", 0.2)
                X_train, X_test, y_train, y_test = dm.train_test_split(test_size=ratio)
                print(f"Данные загружены. Признаков: {X_train.shape[0]}. Обучающих примеров: {X_train.shape[1]}.")
            except (FileNotFoundCustomError, InvalidCSVFormatError, InvalidSplitRatioError) as e:
                print(f"Ошибка: {e}")

        elif choice == '3':
            if nn is None:
                print("Сначала создайте сеть (пункт 1).")
                continue
            if X_train is None:
                print("Сначала загрузите данные (пункт 2).")
                continue
                
            lr = get_float("Скорость обучения (learning rate) [0.01]: ", 0.01)
            epochs = get_int("Число эпох [1000]: ", 1000)
            batch_size = get_int("Размер батча (0 - весь набор) [0]: ", 0)
            if batch_size == 0:
                batch_size = None
                
            nn.train(X_train, y_train, epochs, lr, batch_size)
            print(f"Обучение завершено. Финальная ошибка (MSE): {nn.history['loss'][-1]:.6f}")

        elif choice == '4':
            if nn is None:
                print("Сначала создайте сеть (пункт 1).")
                continue
            
            mode = input("Предсказать для одного примера (1) или из файла (2)? ")
            if mode == '1':
                try:
                    assert X_train is not None
                    raw = input(f"Введите {X_train.shape[0]} признаков через запятую: ")
                    x = np.array([float(v) for v in raw.split(',')]).reshape(-1, 1)
                    # Применяем ту же нормализацию, что и к обучающим данным
                    assert dm.feature_mean is not None
                    assert dm.feature_std is not None
                    x_norm = (x - dm.feature_mean.reshape(-1, 1)) / dm.feature_std.reshape(-1, 1)
                    pred = nn.predict(x_norm)
                    print(f"Предсказание: {pred.item():.4f}")
                except ValueError:
                    print("Неверный ввод признаков.")
                except MismatchedDataError as e:
                    print(f"Ошибка: {e}")
            elif mode == '2':
                try:
                    path = input("Путь к CSV с примерами (только признаки, без целевой переменной): ")
                    df = turtle.pd.read_csv(path)
                    x = df.values.T
                    assert dm.feature_mean is not None
                    assert dm.feature_std is not None
                    x_norm = (x - dm.feature_mean.reshape(-1, 1)) / dm.feature_std.reshape(-1, 1)
                    preds = nn.predict(x_norm)
                    print("Предсказания:", preds.flatten())
                except Exception as e:
                    print(f"Ошибка: {e}")

        elif choice == '5':
            if nn is None:
                print("Сеть не создана.")
                continue
            path = input("Имя файла для сохранения (например, weights.json): ")
            try:
                nn.save_weights(path)
                print("Веса сохранены.")
            except Exception as e:
                print(f"Ошибка: {e}")

        elif choice == '6':
            if nn is None:
                print("Сначала создайте сеть с нужной архитектурой (пункт 1).")
                continue
            path = input("Путь к файлу весов: ")
            try:
                nn.load_weights(path)
                print("Веса загружены.")
            except (FileNotFoundError, MismatchedDataError) as e:
                print(f"Ошибка: {e}")

        elif choice == '7':
            if not nn or not nn.history['loss']:
                print("Нет данных для построения графика. Обучите сеть.")
                continue
            plt.figure(figsize=(10, 6))
            plt.plot(nn.history['loss'], label='MSE Loss')
            plt.title('История ошибки обучения')
            plt.xlabel('Эпоха')
            plt.ylabel('Ошибка (MSE)')
            plt.yscale('log') # Логарифмическая шкала для лучшей визуализации затухания ошибки
            plt.grid(True)
            plt.legend()
            plt.show()

        elif choice == '8':
            print("Выход.")
            break
            
        else:
            print("Неверный пункт меню.")

if __name__ == "__main__":
    main()