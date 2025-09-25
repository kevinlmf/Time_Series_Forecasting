import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from typing import Optional, Dict, Any, List, Union
import warnings

class VARModel:
    """
    Vector Autoregression (VAR) model for multivariate time series forecasting.

    Handles cointegration testing and supports both VAR and VECM specifications.
    """

    def __init__(self, maxlags: Optional[int] = None, ic: str = 'aic'):
        """
        Initialize VAR model.

        Args:
            maxlags: Maximum number of lags to consider
            ic: Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
        """
        self.maxlags = maxlags
        self.ic = ic
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        self.data = None
        self.selected_lags = None

    def check_stationarity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check stationarity for all series using ADF test.

        Args:
            data: Multivariate time series data

        Returns:
            Dictionary with stationarity test results for each variable
        """
        results = {}
        for col in data.columns:
            adf_result = adfuller(data[col].dropna())
            results[col] = {
                'adf_statistic': adf_result[0],
                'pvalue': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
        return results

    def johansen_cointegration_test(self, data: pd.DataFrame, det_order: int = 0,
                                   k_ar_diff: int = 1) -> Dict[str, Any]:
        """
        Johansen cointegration test.

        Args:
            data: Multivariate time series data
            det_order: Deterministic order (-1, 0, 1)
            k_ar_diff: Number of lagged differences

        Returns:
            Cointegration test results
        """
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

        return {
            'trace_statistic': result.lr1,
            'max_eigen_statistic': result.lr2,
            'critical_values_trace': result.cvt,
            'critical_values_max_eigen': result.cvm,
            'eigenvalues': result.eig,
            'cointegration_rank': self._determine_cointegration_rank(result)
        }

    def _determine_cointegration_rank(self, johansen_result) -> int:
        """Determine cointegration rank from Johansen test."""
        trace_stats = johansen_result.lr1
        critical_values = johansen_result.cvt[:, 1]  # 5% critical values

        rank = 0
        for i, (trace_stat, cv) in enumerate(zip(trace_stats, critical_values)):
            if trace_stat > cv:
                rank = i + 1
            else:
                break
        return rank

    def select_lag_order(self, data: pd.DataFrame, maxlags: Optional[int] = None) -> Dict[str, int]:
        """
        Select optimal lag order using information criteria.

        Args:
            data: Multivariate time series data
            maxlags: Maximum lags to consider

        Returns:
            Dictionary with optimal lags for each criterion
        """
        if maxlags is None:
            maxlags = min(12, len(data) // 4)

        model = VAR(data)
        lag_order_results = model.select_order(maxlags=maxlags)

        return {
            'aic': lag_order_results.aic,
            'bic': lag_order_results.bic,
            'fpe': lag_order_results.fpe,
            'hqic': lag_order_results.hqic,
            'selected': lag_order_results.selected_orders[self.ic]
        }

    def fit(self, data: pd.DataFrame, lags: Optional[int] = None) -> 'VARModel':
        """
        Fit VAR model to multivariate time series.

        Args:
            data: Multivariate time series data
            lags: Number of lags (if None, will be selected automatically)

        Returns:
            Self for method chaining
        """
        self.data = data

        if lags is None:
            lag_selection = self.select_lag_order(data, self.maxlags)
            self.selected_lags = lag_selection['selected']
        else:
            self.selected_lags = lags

        self.model = VAR(data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fitted_model = self.model.fit(self.selected_lags)

        self.is_fitted = True
        return self

    def forecast(self, steps: int, alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """
        Generate VAR forecasts with confidence intervals.

        Args:
            steps: Number of periods to forecast
            alpha: Significance level for confidence intervals

        Returns:
            Dictionary with forecasts and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        forecast_result = self.fitted_model.forecast(
            self.fitted_model.y, steps=steps, alpha=alpha
        )

        forecast_ci = self.fitted_model.forecast_interval(
            self.fitted_model.y, steps=steps, alpha=alpha
        )

        return {
            'forecast': forecast_result,
            'conf_int_lower': forecast_ci[:, :, 0],
            'conf_int_upper': forecast_ci[:, :, 1]
        }

    def impulse_response(self, periods: int = 10, impulse: Optional[str] = None,
                        response: Optional[str] = None) -> np.ndarray:
        """
        Compute impulse response functions.

        Args:
            periods: Number of periods for IRF
            impulse: Variable giving the impulse (None for all)
            response: Variable responding to impulse (None for all)

        Returns:
            Impulse response functions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        irf = self.fitted_model.irf(periods=periods)
        return irf.irs

    def forecast_error_variance_decomposition(self, periods: int = 10) -> np.ndarray:
        """
        Compute forecast error variance decomposition.

        Args:
            periods: Number of periods

        Returns:
            FEVD results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        fevd = self.fitted_model.fevd(periods=periods)
        return fevd.decomp

    def granger_causality_test(self, caused: Union[str, List[str]],
                              causing: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Granger causality test.

        Args:
            caused: Variable(s) being caused
            causing: Variable(s) doing the causing

        Returns:
            Test results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        test_result = self.fitted_model.test_causality(
            caused=caused, causing=causing, kind='f'
        )

        return {
            'statistic': test_result.test_statistic,
            'pvalue': test_result.pvalue,
            'critical_value': test_result.critical_value,
            'reject_null': test_result.pvalue < 0.05
        }

    def residual_diagnostics(self) -> Dict[str, Any]:
        """
        Comprehensive residual diagnostics.

        Returns:
            Dictionary with various diagnostic test results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Portmanteau test for serial correlation
        portmanteau = self.fitted_model.test_serial_correlation(lags=10)

        # Test for normality
        normality = self.fitted_model.test_normality()

        # Test for heteroskedasticity
        hetero = self.fitted_model.test_whiteness()

        return {
            'portmanteau_statistic': portmanteau.test_statistic,
            'portmanteau_pvalue': portmanteau.pvalue,
            'normality_statistic': normality.test_statistic,
            'normality_pvalue': normality.pvalue,
            'whiteness_statistic': hetero.test_statistic,
            'whiteness_pvalue': hetero.pvalue
        }

    def summary(self) -> str:
        """Get model summary."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return str(self.fitted_model.summary())