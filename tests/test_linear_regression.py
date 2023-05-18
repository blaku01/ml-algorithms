import numpy as np
from algorithm_implementations.linear_regression import LinearRegression


def test_linear_regression_fit():
    # Test data
    X = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
        ]
    )
    Y = np.array(
        [
            17.83676188,
            21.53365087,
            26.56245756,
            26.51218615,
            36.84499599,
            30.89892445,
            34.99920586,
            39.36864064,
            48.23768354,
            50.90615844,
            56.47300015,
            66.19894943,
            58.32010356,
            72.09500767,
            69.4350343,
            77.45492329,
            73.23941373,
            84.81960418,
            88.51289245,
            88.71126779,
        ]
    )

    # Create a LinearRegression instance
    lr = LinearRegression()

    # Fit the model
    lr.fit(X, Y)

    # Assert that the coefficients are calculated correctly
    expected_a = np.array([12.600780207631601, 3.8902155132255665])
    np.testing.assert_array_equal(lr.a, expected_a)


def test_linear_regression_predict():
    # Test data
    X = np.array([21.0, 22.0, 23.0])

    # Create a LinearRegression instance
    lr = LinearRegression()

    # Set coefficients manually for testing
    lr.a = np.array([12.600780207631601, 3.8902155132255665])

    # Predict the target values
    predicted_Y = lr.predict(X)

    # Assert that the predictions match the expected values
    expected_Y = np.array([94.2953059853685, 98.18552149859406, 102.07573701181963])
    np.testing.assert_array_equal(predicted_Y, expected_Y)
