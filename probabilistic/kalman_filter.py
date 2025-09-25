import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
from filterpy.kalman import ExtendedKalmanFilter as FilterPyExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from typing import Tuple, Optional, Callable, Dict, Any
import matplotlib.pyplot as plt

class KalmanFilter:
    """
    Kalman Filter for linear state-space models.

    Suitable for time series forecasting with linear dynamics.
    """

    def __init__(self, dim_x: int, dim_z: int):
        """
        Initialize Kalman Filter.

        Args:
            dim_x: Dimension of state vector
            dim_z: Dimension of measurement vector
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.kf = FilterPyKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.states = []
        self.covariances = []
        self.measurements = []
        self.is_initialized = False

    def setup_constant_velocity_model(self, dt: float = 1.0, q_var: float = 0.1):
        """
        Setup a constant velocity model for time series forecasting.

        Args:
            dt: Time step
            q_var: Process noise variance
        """
        # State transition matrix (position and velocity)
        self.kf.F = np.array([[1., dt],
                             [0., 1.]])

        # Measurement function (observe position only)
        self.kf.H = np.array([[1., 0.]])

        # Process noise covariance
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=q_var)

        # Measurement noise covariance
        self.kf.R = np.array([[1.0]])

        # Initial state covariance
        self.kf.P *= 100

        self.is_initialized = True

    def setup_local_level_model(self, q_var: float = 0.1, r_var: float = 1.0):
        """
        Setup a local level model (random walk with noise).

        Args:
            q_var: Process noise variance
            r_var: Measurement noise variance
        """
        # State transition matrix (random walk)
        self.kf.F = np.array([[1.0]])

        # Measurement function
        self.kf.H = np.array([[1.0]])

        # Process noise covariance
        self.kf.Q = np.array([[q_var]])

        # Measurement noise covariance
        self.kf.R = np.array([[r_var]])

        # Initial state covariance
        self.kf.P = np.array([[100.0]])

        self.is_initialized = True

    def setup_local_linear_trend_model(self, q_level: float = 0.1,
                                     q_trend: float = 0.01, r_var: float = 1.0):
        """
        Setup a local linear trend model.

        Args:
            q_level: Level process noise variance
            q_trend: Trend process noise variance
            r_var: Measurement noise variance
        """
        # State transition matrix [level, trend]
        self.kf.F = np.array([[1., 1.],
                             [0., 1.]])

        # Measurement function (observe level only)
        self.kf.H = np.array([[1., 0.]])

        # Process noise covariance
        self.kf.Q = np.array([[q_level, 0.],
                             [0., q_trend]])

        # Measurement noise covariance
        self.kf.R = np.array([[r_var]])

        # Initial state covariance
        self.kf.P = np.array([[100., 0.],
                             [0., 100.]])

        self.is_initialized = True

    def fit(self, data: pd.Series, initial_state: Optional[np.ndarray] = None) -> 'KalmanFilter':
        """
        Fit Kalman filter to time series data.

        Args:
            data: Time series data
            initial_state: Initial state estimate

        Returns:
            Self for method chaining
        """
        if not self.is_initialized:
            raise ValueError("Must setup model before fitting")

        # Initialize state
        if initial_state is not None:
            self.kf.x = initial_state
        else:
            self.kf.x = np.zeros(self.dim_x)

        # Reset storage
        self.states = []
        self.covariances = []
        self.measurements = []

        # Filter through data
        for measurement in data.values:
            self.kf.predict()
            self.kf.update(measurement)

            # Store results
            self.states.append(self.kf.x.copy())
            self.covariances.append(self.kf.P.copy())
            self.measurements.append(measurement)

        return self

    def forecast(self, steps: int, return_std: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate forecasts using Kalman filter.

        Args:
            steps: Number of steps to forecast
            return_std: Whether to return prediction standard deviation

        Returns:
            Dictionary with forecasts and uncertainties
        """
        if not self.states:
            raise ValueError("Must fit model before forecasting")

        # Start from last state
        x = self.states[-1].copy()
        P = self.covariances[-1].copy()

        forecasts = []
        forecast_stds = []

        for _ in range(steps):
            # Predict next state
            x = self.kf.F @ x
            P = self.kf.F @ P @ self.kf.F.T + self.kf.Q

            # Predicted observation
            y_pred = self.kf.H @ x
            pred_var = self.kf.H @ P @ self.kf.H.T + self.kf.R

            forecasts.append(y_pred[0] if len(y_pred) == 1 else y_pred)
            if return_std:
                forecast_stds.append(np.sqrt(pred_var[0, 0]) if pred_var.shape == (1, 1) else np.sqrt(np.diag(pred_var)))

        result = {'forecast': np.array(forecasts)}
        if return_std:
            result['forecast_std'] = np.array(forecast_stds)

        return result

    def get_filtered_states(self) -> np.ndarray:
        """Get filtered state estimates."""
        return np.array(self.states)

    def get_smoothed_states(self) -> np.ndarray:
        """Get smoothed state estimates using RTS smoother."""
        from filterpy.kalman import rts_smoother

        # Convert measurements to array
        measurements = np.array(self.measurements).reshape(-1, 1)

        # Run RTS smoother
        smoothed_states, _ = rts_smoother(
            measurements, self.kf.F, self.kf.H, self.kf.Q, self.kf.R,
            np.zeros(self.dim_x), np.eye(self.dim_x) * 100
        )

        return smoothed_states

    def log_likelihood(self) -> float:
        """Calculate log likelihood of the data."""
        if not hasattr(self.kf, 'log_likelihood'):
            return np.nan

        return self.kf.log_likelihood

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state-space models.

    Suitable for time series with nonlinear dynamics.
    """

    def __init__(self, dim_x: int, dim_z: int, hx: Callable, HJacobian: Callable,
                 fx: Callable, FJacobian: Callable):
        """
        Initialize Extended Kalman Filter.

        Args:
            dim_x: Dimension of state vector
            dim_z: Dimension of measurement vector
            hx: Measurement function h(x)
            HJacobian: Jacobian of measurement function
            fx: State transition function f(x, dt)
            FJacobian: Jacobian of state transition function
        """
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.ekf = FilterPyExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)

        # Set user-defined functions
        self.ekf.f = fx
        self.ekf.F = FJacobian
        self.ekf.h = hx
        self.ekf.H = HJacobian

        self.states = []
        self.covariances = []
        self.measurements = []

    def setup_process_noise(self, Q: np.ndarray):
        """Set process noise covariance matrix."""
        self.ekf.Q = Q

    def setup_measurement_noise(self, R: np.ndarray):
        """Set measurement noise covariance matrix."""
        self.ekf.R = R

    def fit(self, data: pd.Series, initial_state: np.ndarray,
            initial_covariance: np.ndarray, dt: float = 1.0) -> 'ExtendedKalmanFilter':
        """
        Fit Extended Kalman filter to time series data.

        Args:
            data: Time series data
            initial_state: Initial state estimate
            initial_covariance: Initial state covariance
            dt: Time step

        Returns:
            Self for method chaining
        """
        # Initialize
        self.ekf.x = initial_state
        self.ekf.P = initial_covariance

        # Reset storage
        self.states = []
        self.covariances = []
        self.measurements = []

        # Filter through data
        for measurement in data.values:
            self.ekf.predict(dt=dt)
            self.ekf.update(measurement)

            # Store results
            self.states.append(self.ekf.x.copy())
            self.covariances.append(self.ekf.P.copy())
            self.measurements.append(measurement)

        return self

    def forecast(self, steps: int, dt: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate forecasts using Extended Kalman filter.

        Args:
            steps: Number of steps to forecast
            dt: Time step

        Returns:
            Dictionary with forecasts
        """
        if not self.states:
            raise ValueError("Must fit model before forecasting")

        # Start from last state
        x = self.states[-1].copy()
        P = self.covariances[-1].copy()

        forecasts = []
        forecast_stds = []

        for _ in range(steps):
            # Predict next state
            x = self.ekf.f(x, dt)
            F = self.ekf.F(x)
            P = F @ P @ F.T + self.ekf.Q

            # Predicted observation
            y_pred = self.ekf.h(x)
            H = self.ekf.H(x)
            pred_var = H @ P @ H.T + self.ekf.R

            forecasts.append(y_pred[0] if len(y_pred) == 1 else y_pred)
            forecast_stds.append(np.sqrt(pred_var[0, 0]) if pred_var.shape == (1, 1) else np.sqrt(np.diag(pred_var)))

        return {
            'forecast': np.array(forecasts),
            'forecast_std': np.array(forecast_stds)
        }