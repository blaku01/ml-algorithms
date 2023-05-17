import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """Linear regression is a supervised machine learning algorithm used for
    predicting a continuous target variable based on one or more independent
    features.

    This class provides an implementation of the basic Linear Regression
    algorithm.

    Example usage:
    >>> X = np.array([[1], [2], [3]])
    >>> Y = np.array([[3], [4], [5]])
    >>> lr = LinearRegression()
    >>> lr.fit(X, Y)
    >>> predictions = lr.predict(X)
    >>> print(predictions)
    [[ 3.]
     [ 4.]
     [ 5.]]
    """

    def __init__(self) -> None:
        """Initialize a LinearRegression instance.

        Attributes:
        - a (array-like, shape (1, m)): The coefficient vector of the linear regression model, where m is the number of parameters.
        """
        self.a = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Fit the linear regression model to the given training data.

        Parameters:
        - X (np.ndarray, shape (n, m)): The input features matrix, where n is the number of samples and m is the number of features.
        - Y (np.ndarray, shape (n, 1)): The target values.

        Note: The input features matrix X should have a shape of (n, m), where n is the number of samples and m is the number of features.
        The target values Y should have a shape of (n, 1).

        Returns:
        - self (np.ndarray, shape (1, m)): Returns an instance of the LinearRegression class.
        """
        X_ext = self.add_constant(X)
        self.a = Y.T @ X_ext @ np.linalg.inv(X_ext.T @ X_ext)
        return self.a

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target values based on the given input data.

        Parameters:
        - X (np.ndarray): The input data to predict the target values for. It should have a shape of (n, m),
            where n is the number of samples and m is the number of features.

        Returns:
        - np.ndarray: The predicted target values based on the input data X. It has a shape of (n, 1).

        Note:
        - The input data X should have the same number of features (m) as the data used during training the model.
        - The target values are predicted using the trained coefficients (a) of the linear regression model.
        - If the model has not been trained (i.e., coefficients are not set), the function will throw an error.
        """
        X_ext = self.add_constant(X)
        return X_ext @ self.a

    @staticmethod
    def add_constant(X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones((X.shape[0], 1)), X])

    def visualize(self, X: np.ndarray, Y: np.ndarray = None) -> None:
        """Visualize the predicted values and, optionally, the actual target
        values.

        Parameters:
        - X (np.ndarray): The input data used for visualization. It should have a shape of (n, 1),
        where n is the number of samples.
        - Y (np.ndarray, optional): The actual target values corresponding to the input data X. It should have
        the same shape as X. Default is None.

        Returns:
        - None: This method does not return anything.

        Note:
        - The method plots the predicted values against the input data X using a red line.
        - If the actual target values (Y) are provided, they are plotted as blue scattered points.
        - This method can be used to visualize the results of the linear regression model.
        """
        plt.plot(X, self.predict(X), c='r', label='y_pred')
        if Y is not None:
            plt.scatter(X, Y, c='b', label='y')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    lr = LinearRegression()
    size = 20
    X = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19.,20.])
    Y = np.array([17.83676188,21.53365087,26.56245756,26.51218615,36.84499599,30.89892445,34.99920586,39.36864064,48.23768354,50.90615844,56.47300015,66.19894943,58.32010356,72.09500767,69.4350343,77.45492329,73.23941373,84.81960418,88.51289245,88.71126779])

    X_test = np.array([21., 22., 23.])
    lr.fit(X, Y)
    for i in lr.predict(X_test):
        print(i)

    lr.visualize(X, Y)