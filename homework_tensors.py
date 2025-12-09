import torch



tensor_random = torch.rand(3, 4)
print("Тензор 3x4 со случайными числами от 0 до 1:")
print(tensor_random)

tensor_zeros = torch.zeros(2, 3, 4)
print("\nТензор 2x3x4, заполненный нулями:")
print(tensor_zeros)

tensor_ones = torch.ones(5, 5)
print("\nТензор 5x5, заполненный единицами:")
print(tensor_ones)

tensor_range = torch.arange(16).reshape(4, 4)
print("\nТензор 4x4 с числами от 0 до 15:")
print(tensor_range)

A = torch.rand(3, 4)
B = torch.rand(4, 3)
print("\nТензор A (3x4):")
print(A)
print("\nТензор B (4x3):")
print(B)

A_T = A.T
print("\nТранспонированный A (4x3):")
print(A_T)

matmul_result = torch.matmul(A, B)
print("\nМатричное умножение A @ B (3x3):")
print(matmul_result)

elementwise_mul = A * B.T
print("\nПоэлементное умножение A * B.T (3x4):")
print(elementwise_mul)

sum_A = torch.sum(A)
print(f"\nСумма всех элементов A: {sum_A}")


tensor_5x5x5 = torch.rand(5, 5, 5)
print("\nТензор 5x5x5:")
print(tensor_5x5x5)

first_row = tensor_5x5x5[0, :, :]
print("\nПервая строка (0,:,:):")
print(first_row)

last_column = tensor_5x5x5[:, :, -1]
print("\nПоследний столбец (:,:,-1):")
print(last_column)

center_submatrix = tensor_5x5x5[1:3, 1:3, 1:3]
print("\nПодматрица 2x2x2 из центра:")
print(center_submatrix)

even_indices = tensor_5x5x5[::2, ::2, ::2]
print("\nЭлементы с четными индексами (по всем осям):")
print(even_indices)


tensor_24 = torch.arange(24)
print("\nТензор с 24 элементами:")
print(tensor_24)

shape_2x12 = tensor_24.reshape(2, 12)
print("\nФорма 2x12:")
print(shape_2x12)

shape_3x8 = tensor_24.reshape(3, 8)
print("\nФорма 3x8:")
print(shape_3x8)

shape_4x6 = tensor_24.reshape(4, 6)
print("\nФорма 4x6:")
print(shape_4x6)

shape_2x3x4 = tensor_24.reshape(2, 3, 4)
print("\nФорма 2x3x4:")
print(shape_2x3x4)

shape_2x2x2x3 = tensor_24.reshape(2, 2, 2, 3)
print("\nФорма 2x2x2x3:")
print(shape_2x2x2x3)