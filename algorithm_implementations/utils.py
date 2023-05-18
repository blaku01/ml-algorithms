import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 1],
              [1, 4],
              [1, 8],
              [1, 20]])

Y = np.array([[2], [8], [16], [40]]) + np.ones((4,1)) * 5

# Current estimation
f = np.array([[0], [1]])

# Define the learning rate and the number of iterations
alpha = 0.01
num_iterations = 1000

# Perform gradient descent
for _ in range(num_iterations):
    predictions = X @ f
    loss = predictions - Y
    import numpy as np

X = np.array([[1, 1],
              [1, 4],
              [1, 8],
              [1, 20]])

Y = np.array([[2], [8], [16], [40]]) + np.ones((4, 1)) * 5

# Current estimation
f = np.array([[0], [1]])

# Define the learning rate and the number of iterations
alpha = 0.001
num_iterations = 1000

# Perform gradient descent
for _ in range(num_iterations):
    predictions = X @ f
    loss = predictions - Y

    # Calculate the gradients
    gradients = X.T @ loss * 1/X.shape[0]

    # Update the estimation
    f = f - alpha * gradients

print((X @ f).reshape(-1))
plt.scatter(X[:, 1], Y, c='b', label="actual data")
plt.plot(X[:, 1], (X @ f).reshape(-1), c='r', label=f'predicted f(x) = {f[1]} x + {f[0]}')
plt.legend()
plt.show()
