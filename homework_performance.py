import torch
import time


sizes = [
    (64, 1024, 1024),
    (128, 512, 512),
    (256, 256, 256)
]

matrices = []
for size in sizes:
    matrix = torch.rand(size)
    matrices.append(matrix)
    print(f"Создана матрица размера {size}")


def measure_time_cpu(func, *args, num_runs=5):
    """
    Измеряет время выполнения функции на CPU.

    Args:
        func: Функция для измерения
        *args: Аргументы функции
        num_runs: Количество запусков для усреднения

    Returns:
        float: Среднее время в миллисекундах
    """
    times = []
    for _ in range(num_runs):
        start = time.time()
        result = func(*args)
        end = time.time()
        times.append((end - start) * 1000)
    return sum(times) / len(times), result

def measure_time_gpu(func, *args, num_runs=5):
    """
    Измеряет время выполнения функции на GPU.

    Args:
        func: Функция для измерения
        *args: Аргументы функции
        num_runs: Количество запусков для усреднения

    Returns:
        float: Среднее время в миллисекундах
    """
    if not torch.cuda.is_available():
        return None, None

    gpu_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            gpu_args.append(arg.cuda())
        else:
            gpu_args.append(arg)

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()  
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = func(*gpu_args)
        end.record()
        torch.cuda.synchronize()  
        times.append(start.elapsed_time(end))
    return sum(times) / len(times), result


operations = [
    ("Матричное умножение", lambda a, b: torch.matmul(a, b)),
    ("Поэлементное сложение", lambda a, b: a + b),
    ("Поэлементное умножение", lambda a, b: a * b),
    ("Транспонирование", lambda a, b: a.T),
    ("Вычисление суммы", lambda a, b: torch.sum(a))
]

results = []
for i, matrix in enumerate(matrices):
    size = sizes[i]
    print(f"\nМатрица размера {size}:")

    for op_name, op_func in operations:
        if op_name in ["Матричное умножение", "Поэлементное сложение", "Поэлементное умножение"]:
            matrix2 = torch.rand(size)
            args = (matrix, matrix2)
        else:
            args = (matrix, None)

        cpu_time, _ = measure_time_cpu(op_func, *args)

        gpu_time, _ = measure_time_gpu(op_func, *args)

        # Вычисление ускорения
        if gpu_time is not None:
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            speedup_str = f"{speedup:.1f}x"
        else:
            speedup_str = "N/A"

        results.append({
            'size': size,
            'operation': op_name,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup_str
        })

        print(f"  {op_name}: CPU {cpu_time:.2f}мс, GPU {gpu_time:.2f}мс, Ускорение {speedup_str}")

print(f"{'Размер матрицы':<20} {'Операция':<25} {'CPU (мс)':<10} {'GPU (мс)':<10} {'Ускорение':<10}")
print("-"*80)

for result in results:
    size_str = f"{result['size'][0]}x{result['size'][1]}x{result['size'][2]}"
    gpu_time_str = f"{result['gpu_time']:.2f}" if result['gpu_time'] is not None else "N/A"
    print(f"{size_str:<20} {result['operation']:<25} {result['cpu_time']:<10.2f} {gpu_time_str:<10} {result['speedup']:<10}")

