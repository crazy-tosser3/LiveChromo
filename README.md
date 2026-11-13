# LiveChromo

Приложение для замены фона на видео с камеры или из файла. Сегментация выполняется нейросетевой ONNX-моделью, обработка видео — через OpenCV, inference — через ONNX Runtime.

Поддерживаются:

- CPU;
- NVIDIA GPU через CUDA Execution Provider;
- Apple GPU через CoreML Execution Provider, использующий Metal;
- автоматический выбор доступного ускорителя.

## Требования

- CMake 3.20 или новее;
- компилятор с поддержкой C++17;
- OpenCV с модулями `core`, `imgproc`, `imgcodecs`, `highgui` и `videoio`;
- ONNX Runtime;
- ONNX-модель сегментации `model.onnx`.

```
mkdir -p third_party

curl -fL "https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-osx-arm64-1.31.1.tgz" -o onnxruntime.tgz

tar -xzf onnxruntime.tgz mv onnxruntime-osx-arm64-1.31.1 third_party/onnxruntime rm onnxruntime.tgz
```

В каталоге `third_party/onnxruntime` уже находится macOS arm64-сборка ONNX Runtime 1.22.0 с поддержкой CoreML. Для другой платформы или CUDA укажите путь к подходящей сборке через `ONNXRUNTIME_ROOT`.

## Сборка

### Release на macOS

Release рекомендуется для реальной работы: Debug заметно замедляет OpenCV и обработку кадров.

```bash
cmake -S . -B cmake-build-release \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --target segmentation -j 4
```

Если CMake не находит OpenCV, укажите каталог с `OpenCVConfig.cmake`:

```bash
cmake -S . -B cmake-build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
```

При использовании vcpkg также можно передать его toolchain:

```bash
cmake -S . -B cmake-build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### Сборка с поддержкой CUDA

Нужны NVIDIA GPU, драйвер NVIDIA, CUDA/cuDNN и ONNX Runtime, собранный с CUDA Execution Provider. Обычная CPU-сборка ONNX Runtime не поддерживает CUDA.

```bash
cmake -S . -B cmake-build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-gpu
cmake --build cmake-build-release --target segmentation -j 4
```

По умолчанию используется CUDA-устройство `0`.

## Запуск

Камера 1 и автоматический выбор устройства:

```bash
./cmake-build-release/segmentation --model model.onnx
```

Явный выбор устройства:

```bash
# Apple GPU через CoreML/Metal
./cmake-build-release/segmentation --model model.onnx --device metal

# NVIDIA GPU
./cmake-build-release/segmentation --model model.onnx --device cuda

# CPU
./cmake-build-release/segmentation --model model.onnx --device cpu
```

Камера с изображением фона:

```bash
./cmake-build-release/segmentation \
  --model model.onnx \
  --camera 1 \
  --bg background.jpg \
  --device auto
```

Обработка видеофайла и сохранение результата:

```bash
./cmake-build-release/segmentation \
  --model model.onnx \
  --video input.mp4 \
  --bg background.jpg \
  --out result.mp4 \
  --device auto
```

При использовании `--out` окна не открываются. Рекомендуется указывать выходной файл с расширением `.mp4`.

В оконном режиме для выхода нажмите `Q` или `Esc`.

## Аргументы командной строки

| Аргумент | Описание |
|---|---|
| `--camera N` | Индекс камеры. По умолчанию `1`. |
| `--video FILE` | Использовать видеофайл вместо камеры. |
| `--bg IMAGE` | Изображение фона. Без него используется зелёный фон. |
| `--model FILE` | Путь к ONNX-модели. По умолчанию `../model.onnx`. |
| `--out FILE` | Сохранить результат в файл без открытия окон. |
| `--device DEVICE` | Устройство inference: `auto`, `cpu`, `cuda` или `metal`. По умолчанию `auto`. |
| `--help`, `-h` | Показать краткую справку. |

Поддерживается и сокращённый позиционный формат:

```bash
./cmake-build-release/segmentation input.mp4 background.jpg --model model.onnx
```

## Выбор устройства

### `auto`

- На Apple выбирается CoreML/Metal, если provider доступен, иначе CPU.
- На остальных платформах выбирается CUDA, если provider доступен, иначе CPU.

Режим `auto` проверяет наличие provider в сборке ONNX Runtime, но не измеряет, какой provider быстрее для конкретной модели.

### `metal`

В ONNX Runtime нет отдельного Metal Execution Provider. Режим `metal` использует CoreML с compute units `CPUAndGPU`. Поддерживаемые CoreML операции могут выполняться на Apple GPU через Metal, остальные операции остаются на CPU.

Предупреждения ONNX Runtime о части узлов, не назначенных CoreML, ожидаемы для такой модели и не означают, что Metal полностью отключён.

### `cuda`

Использует CUDA Execution Provider V2. Если появляется сообщение `CUDA is unavailable in this ONNX Runtime build`, необходимо указать путь к CUDA-сборке ONNX Runtime через `ONNXRUNTIME_ROOT` и заново выполнить конфигурацию CMake.

## Производительность на Apple Silicon

Для текущей модели на Apple M3 CoreML разбивает граф на несколько участков, поэтому Metal может работать медленнее CPU. В тесте на 60 кадрах 1280×720 с сохранением результата получены следующие значения:

| Конфигурация | Metal | CPU |
|---|---:|---:|
| Debug | 16,96 с | 16,33 с |
| Release | 4,63 с | 3,42 с |

Результаты зависят от модели, разрешения и нагрузки на систему. Для этой модели на M3 рекомендуется начать с Release и `--device cpu`, затем сравнить с `--device metal`.

## Обработка кадра

1. Кадр масштабируется до 256×256, переводится в RGB и нормализуется к диапазону `[0, 1]`.
2. ONNX-модель вычисляет альфа-маску.
3. Маска сглаживается во времени, оставляется крупнейшая связная компонента и применяется пространственное сглаживание.
4. Исходный кадр и фон смешиваются по альфа-маске.

Inference выполняется в отдельном потоке. В интерактивном режиме отображается самая свежая готовая маска; при сохранении через `--out` кадры обрабатываются последовательно, чтобы не пропускать результаты.
