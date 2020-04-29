import random
from linear_algebra import  add, scalar_multiply, distance

def gradient_step(v, gradient, step_size):
    assert (len(v) == len(gradient))
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v):
    return [2* v_i for v_i in v]

v =[random.uniform(-10, 10) for i in range(3)]


for epoch in range(1000):
    grad = sum_of_squares_gradient(v)
    v = gradient_step(v, grad, -0.01)
    print(epoch, v)


