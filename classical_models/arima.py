import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Tuple, Optional, Dict, Any
import warnings

class ARIMAModel:
    """
    ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting.

    Supports automatic order selection and seasonal decomposition.
    """

    def __init__(self, order: Optional[Tuple[int, int, int]] = None,
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                 auto_order: bool = True):
        """
        Initialize ARIMA model.

        Args:
            order: (p, d, q) parameters for ARIMA
            seasonal_order: (P, D, Q, s) parameters for seasonal ARIMA
            auto_order: Whether to automatically determine optimal order
        """
        self.order = order
        # Set seasonal_order to None explicitly for non-seasonal ARIMA
        self.seasonal_order = seasonal_order
        self.auto_order = auto_order
        self.model = None
        self.fitted_model = None
        self.is_fitted = False

    def check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Check stationarity using ADF and KPSS tests.

        Args:
            series: Time series data

        Returns:
            Dictionary with test results
        """
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna())

        # KPSS test
        kpss_result = kpss(series.dropna())

        return {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_critical_values': adf_result[4],
            'is_stationary_adf': adf_result[1] < 0.05,
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'kpss_critical_values': kpss_result[3],
            'is_stationary_kpss': kpss_result[1] > 0.05
        }

    def auto_select_order(self, series: pd.Series, max_p: int = 5,
                         max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Automatically select optimal ARIMA order using AIC.

        Args:
            series: Time series data
            max_p, max_d, max_q: Maximum values for p, d, q parameters

        Returns:
            Optimal (p, d, q) order
        """
        best_aic = np.inf
        best_order = None

        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()

                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)

                    except Exception:
                        continue

        return best_order if best_order else (1, 1, 1)

    def fit(self, series: pd.Series) -> 'ARIMAModel':
        """
        Fit ARIMA model to time series data.

        Args:
            series: Time series data

        Returns:
            Self for method chaining
        """
        if self.auto_order and self.order is None:
            self.order = self.auto_select_order(series)

        # Ensure we have a valid order
        if self.order is None:
            self.order = (1, 1, 1)  # Default order

        # Create ARIMA model - only include seasonal_order if it's not None
        if self.seasonal_order is not None:
            self.model = ARIMA(series, order=self.order,
                              seasonal_order=self.seasonal_order)
        else:
            self.model = ARIMA(series, order=self.order)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted_model = self.model.fit()

        self.is_fitted = True
        return self

    def forecast(self, steps: int, alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """
        Generate forecasts with confidence intervals.

        Args:
            steps: Number of periods to forecast
            alpha: Significance level for confidence intervals

        Returns:
            Dictionary with forecasts and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        forecast_result = self.fitted_model.get_forecast(steps=steps, alpha=alpha)

        conf_int = forecast_result.conf_int()

        return {
            'forecast': forecast_result.predicted_mean.values,
            'conf_int_lower': conf_int.iloc[:, 0].values,
            'conf_int_upper': conf_int.iloc[:, 1].values,
            'se': getattr(forecast_result, 'se', forecast_result.predicted_mean * 0 + 1).values
        }

    def get_residuals(self) -> pd.Series:
        """Get model residuals."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.resid

    def diagnostic_plots(self):
        """Generate diagnostic plots for model validation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.plot_diagnostics(figsize=(12, 8))

    def summary(self) -> str:
        """Get model summary statistics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.summary()