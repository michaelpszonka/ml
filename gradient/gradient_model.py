from linear_algebra import vector_mean, add, scalar_multiply
import random

#generate random parameters to begin model fit
def parameter_generation():
    params = [(x, 20 * x + 5) for x in range(-50, 50)]
    return params


#loss function for minimzing the squared errors
def linear_gradient(x, y, theta):
    slope, y_intercept = theta
    prediction = slope * x + y_intercept
    error = prediction - y
    squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad

#moving vector v in the direction of the gradient by the learning rate
def gradient_step(v, gradient, step_size):
    assert (len(v) == len(gradient))
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


#fit model to minimize squared errors
def fit_model(parameters):
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

    learning_rate = 0.001
    for epoch in range(5000):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in parameters])
        theta = gradient_step(theta, grad, -learning_rate)     #take step in direction of gradient
        print(epoch, theta)

    slope, intercept = theta
    assert 19.9 < slope < 20.1 # slope should be around 20
    assert 4.9 < intercept < 5.1 # intercept should be around 5

if __name__ == '__main__':
    params = parameter_generation()
    fit_model(params)
