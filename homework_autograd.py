import torch


# Создайте тензоры x, y, z с requires_grad=True
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

f = x**2 + y**2 + z**2 + 2*x*y*z
print(f"Функция f(x,y,z) = {f}")

f.backward()

print(f"Градиент по x: {x.grad}")
print(f"Градиент по y: {y.grad}")
print(f"Градиент по z: {z.grad}")


def mse_loss(y_pred, y_true):
    """
    Вычисляет Mean Squared Error (MSE).

    Args:
        y_pred (torch.Tensor): Предсказанные значения
        y_true (torch.Tensor): Истинные значения

    Returns:
        torch.Tensor: Значение MSE
    """
    return torch.mean((y_pred - y_true)**2)


x = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
y_true = torch.tensor([2.0, 4.0, 6.0], requires_grad=False)


w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)


y_pred = w * x + b


loss = mse_loss(y_pred, y_true)
print(f"\nMSE Loss: {loss}")


loss.backward()

print(f"Градиент по w: {w.grad}")
print(f"Градиент по b: {b.grad}")


x = torch.tensor(1.0, requires_grad=True)
f = torch.sin(x**2 + 1)
print(f"\nФункция f(x) = sin(x^2 + 1) при x=1: {f}")

f.backward()
print(f"Градиент df/dx: {x.grad}")

x = torch.tensor(1.0, requires_grad=True)
f = torch.sin(x**2 + 1)
grad = torch.autograd.grad(f, x)[0]
print(f"Проверка с torch.autograd.grad: {grad}")
