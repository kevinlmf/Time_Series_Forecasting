"""
Stock Market Forecasting Application

Comprehensive example for stock return and volatility forecasting using various models
from the TimeSeries_Forecasting framework.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import framework components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical_models.arima import ARIMAModel
from classical_models.garch import GARCHModel
from deep_learning.rnn_models import LSTMForecaster, RNNTrainer
from evaluation.metrics import ForecastMetrics

class StockForecaster:
    """
    Comprehensive stock forecasting pipeline with multiple models.
    """

    def __init__(self, ticker: str, period: str = '2y'):
        """
        Initialize stock forecaster.

        Args:
            ticker: Stock ticker symbol
            period: Data period ('1y', '2y', '5y', etc.)
        """
        self.ticker = ticker
        self.period = period
        self.data = None
        self.returns = None
        self.models = {}
        self.forecasts = {}
        self.evaluation_results = {}

    def load_data(self) -> pd.DataFrame:
        """
        Load stock data from Yahoo Finance.

        Returns:
            Stock price DataFrame
        """
        print(f"Loading data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=self.period)

        # Calculate returns
        self.returns = np.log(self.data['Close']).diff().dropna()

        print(f"Loaded {len(self.data)} days of data")
        return self.data

    def prepare_train_test_split(self, test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            test_size: Proportion of data for testing

        Returns:
            Training and testing data
        """
        split_point = int(len(self.returns) * (1 - test_size))
        train_data = self.returns[:split_point]
        test_data = self.returns[split_point:]

        print(f"Train set: {len(train_data)} samples")
        print(f"Test set: {len(test_data)} samples")

        return train_data, test_data

    def fit_arima_model(self, train_data: pd.Series) -> Dict[str, Any]:
        """
        Fit ARIMA model for return forecasting.

        Args:
            train_data: Training data

        Returns:
            Model fitting results
        """
        print("Fitting ARIMA model...")

        # Check stationarity
        arima = ARIMAModel(order=(2, 1, 2), auto_order=False)  # Use fixed parameters for speed
        stationarity = arima.check_stationarity(train_data)

        # Fit model
        arima.fit(train_data)
        self.models['arima'] = arima

        return {
            'model': arima,
            'stationarity_test': stationarity,
            'order': arima.order,
            'aic': arima.fitted_model.aic,
            'bic': arima.fitted_model.bic
        }

    def fit_garch_model(self, train_data: pd.Series) -> Dict[str, Any]:
        """
        Fit GARCH model for volatility forecasting.

        Args:
            train_data: Training data (returns)

        Returns:
            Model fitting results
        """
        print("Fitting GARCH model...")

        garch = GARCHModel(vol='GARCH', p=1, q=1, mean='Constant')
        garch.fit(train_data)
        self.models['garch'] = garch

        # Run diagnostic tests
        arch_test = garch.arch_lm_test(lags=5)
        ljung_box = garch.ljung_box_test(lags=10)

        return {
            'model': garch,
            'parameters': garch.get_parameters(),
            'aic': garch.aic(),
            'bic': garch.bic(),
            'arch_test': arch_test,
            'ljung_box_test': ljung_box
        }

    def fit_lstm_model(self, train_data: pd.Series, seq_length: int = 20) -> Dict[str, Any]:
        """
        Fit LSTM model for return forecasting.

        Args:
            train_data: Training data
            seq_length: Sequence length for LSTM

        Returns:
            Model fitting results
        """
        print("Fitting LSTM model...")

        # Initialize model
        lstm = LSTMForecaster(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout=0.2
        )

        # Train model
        trainer = RNNTrainer(lstm, learning_rate=0.001)
        history = trainer.fit(
            train_data=train_data,
            seq_length=seq_length,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=False
        )

        self.models['lstm'] = trainer

        return {
            'model': trainer,
            'training_history': history,
            'final_train_loss': history['train_losses'][-1],
            'final_val_loss': history['val_losses'][-1]
        }

    def generate_forecasts(self, test_data: pd.Series, horizon: int = 10) -> Dict[str, Any]:
        """
        Generate forecasts using all fitted models.

        Args:
            test_data: Test data for evaluation
            horizon: Forecast horizon

        Returns:
            Dictionary with all forecasts
        """
        print(f"Generating {horizon}-step forecasts...")

        forecasts = {}

        # ARIMA forecasts
        if 'arima' in self.models:
            arima_forecast = self.models['arima'].forecast(steps=horizon)
            forecasts['arima'] = {
                'mean': arima_forecast['forecast'],
                'lower': arima_forecast['conf_int_lower'],
                'upper': arima_forecast['conf_int_upper']
            }

        # GARCH volatility forecasts
        if 'garch' in self.models:
            garch_forecast = self.models['garch'].forecast(horizon=horizon)
            forecasts['garch'] = {
                'volatility': garch_forecast['volatility'],
                'variance': garch_forecast['variance']
            }

        # LSTM forecasts
        if 'lstm' in self.models:
            lstm_forecast = self.models['lstm'].predict(
                self.returns, seq_length=20, steps=horizon
            )
            forecasts['lstm'] = {
                'mean': lstm_forecast.flatten()
            }

        self.forecasts = forecasts
        return forecasts

    def evaluate_forecasts(self, test_data: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate forecast accuracy.

        Args:
            test_data: Actual test data

        Returns:
            Evaluation metrics for each model
        """
        print("Evaluating forecasts...")

        results = {}

        # Evaluate ARIMA
        if 'arima' in self.forecasts:
            arima_pred = self.forecasts['arima']['mean']
            actual_values = test_data.values[:len(arima_pred)]

            results['arima'] = ForecastMetrics.comprehensive_evaluation(
                y_true=actual_values,
                y_pred=arima_pred,
                lower_bound=self.forecasts['arima']['lower'],
                upper_bound=self.forecasts['arima']['upper']
            )

        # Evaluate LSTM
        if 'lstm' in self.forecasts:
            lstm_pred = self.forecasts['lstm']['mean']
            actual_values = test_data.values[:len(lstm_pred)]

            results['lstm'] = ForecastMetrics.comprehensive_evaluation(
                y_true=actual_values,
                y_pred=lstm_pred
            )

        # For GARCH, evaluate volatility forecasting (use realized volatility)
        if 'garch' in self.forecasts:
            # Calculate realized volatility (rolling std)
            realized_vol = test_data.rolling(window=5).std().dropna()
            garch_vol = self.forecasts['garch']['volatility']

            # Match lengths for evaluation
            min_length = min(len(realized_vol), len(garch_vol))

            results['garch'] = ForecastMetrics.comprehensive_evaluation(
                y_true=realized_vol.values[:min_length],
                y_pred=garch_vol[:min_length]
            )

        self.evaluation_results = results
        return results

    def plot_results(self, test_data: pd.Series, save_plots: bool = True):
        """
        Plot forecasting results.

        Args:
            test_data: Test data
            save_plots: Whether to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.ticker} Stock Forecasting Results', fontsize=16)

        # Plot 1: Price and Returns
        axes[0, 0].plot(self.data.index[-100:], self.data['Close'][-100:])
        axes[0, 0].set_title('Stock Price (Last 100 days)')
        axes[0, 0].set_ylabel('Price ($)')

        # Plot 2: Returns and ARIMA forecast
        if 'arima' in self.forecasts:
            test_dates = test_data.index[:len(self.forecasts['arima']['mean'])]
            axes[0, 1].plot(test_data.index[:20], test_data.values[:20],
                           label='Actual', color='blue')
            axes[0, 1].plot(test_dates, self.forecasts['arima']['mean'],
                           label='ARIMA', color='red', linestyle='--')
            axes[0, 1].fill_between(test_dates,
                                   self.forecasts['arima']['lower'],
                                   self.forecasts['arima']['upper'],
                                   alpha=0.3, color='red')
        axes[0, 1].set_title('Returns Forecast (ARIMA)')
        axes[0, 1].legend()

        # Plot 3: LSTM forecast
        if 'lstm' in self.forecasts:
            test_dates = test_data.index[:len(self.forecasts['lstm']['mean'])]
            axes[1, 0].plot(test_data.index[:20], test_data.values[:20],
                           label='Actual', color='blue')
            axes[1, 0].plot(test_dates, self.forecasts['lstm']['mean'],
                           label='LSTM', color='green', linestyle='--')
        axes[1, 0].set_title('Returns Forecast (LSTM)')
        axes[1, 0].legend()

        # Plot 4: Volatility forecast (GARCH)
        if 'garch' in self.forecasts:
            realized_vol = test_data.rolling(window=5).std()
            axes[1, 1].plot(realized_vol.index[:20], realized_vol.values[:20],
                           label='Realized Vol', color='blue')
            axes[1, 1].plot(test_data.index[:len(self.forecasts['garch']['volatility'])],
                           self.forecasts['garch']['volatility'],
                           label='GARCH Vol', color='purple', linestyle='--')
        axes[1, 1].set_title('Volatility Forecast (GARCH)')
        axes[1, 1].legend()

        plt.tight_layout()

        if save_plots:
            plt.savefig(f'{self.ticker}_forecasting_results.png', dpi=300, bbox_inches='tight')

        plt.show()

    def run_complete_analysis(self, test_size: float = 0.2, forecast_horizon: int = 10) -> Dict[str, Any]:
        """
        Run complete forecasting analysis pipeline.

        Args:
            test_size: Test set proportion
            forecast_horizon: Number of periods to forecast

        Returns:
            Complete analysis results
        """
        # Load data
        self.load_data()

        # Split data
        train_data, test_data = self.prepare_train_test_split(test_size)

        # Fit models
        arima_results = self.fit_arima_model(train_data)
        garch_results = self.fit_garch_model(train_data)
        lstm_results = self.fit_lstm_model(train_data)

        # Generate forecasts
        forecasts = self.generate_forecasts(test_data, forecast_horizon)

        # Evaluate models
        evaluation = self.evaluate_forecasts(test_data)

        # Plot results
        self.plot_results(test_data)

        # Print summary
        self.print_summary()

        return {
            'model_fits': {
                'arima': arima_results,
                'garch': garch_results,
                'lstm': lstm_results
            },
            'forecasts': forecasts,
            'evaluation': evaluation
        }

    def print_summary(self):
        """Print analysis summary."""
        print(f"\n{'='*50}")
        print(f"FORECASTING SUMMARY FOR {self.ticker}")
        print(f"{'='*50}")

        # Model comparison
        if self.evaluation_results:
            print("\nModel Performance Comparison:")
            print("-" * 40)

            metrics_to_show = ['RMSE', 'MAE', 'MAPE', 'Directional_Accuracy']

            for metric in metrics_to_show:
                print(f"\n{metric}:")
                for model_name, results in self.evaluation_results.items():
                    if metric in results:
                        value = results[metric]
                        if metric == 'Directional_Accuracy':
                            print(f"  {model_name.upper()}: {value:.2f}%")
                        elif metric == 'MAPE':
                            print(f"  {model_name.upper()}: {value:.2f}%")
                        else:
                            print(f"  {model_name.upper()}: {value:.6f}")

def main():
    """Example usage of the stock forecasting pipeline."""

    # Example 1: Apple stock forecasting
    print("Example 1: Apple Stock Forecasting")
    aapl_forecaster = StockForecaster('AAPL', period='2y')
    aapl_results = aapl_forecaster.run_complete_analysis(
        test_size=0.2,
        forecast_horizon=10
    )

    # Example 2: Tesla stock forecasting
    print("\n\nExample 2: Tesla Stock Forecasting")
    tsla_forecaster = StockForecaster('TSLA', period='2y')
    tsla_results = tsla_forecaster.run_complete_analysis(
        test_size=0.2,
        forecast_horizon=10
    )

    # Compare performance across stocks
    print("\n\nCross-Stock Performance Comparison:")
    print("-" * 50)

    stocks = {'AAPL': aapl_forecaster, 'TSLA': tsla_forecaster}

    for metric in ['RMSE', 'MAPE']:
        print(f"\n{metric}:")
        for stock_name, forecaster in stocks.items():
            print(f"{stock_name}:")
            for model_name, results in forecaster.evaluation_results.items():
                if metric in results:
                    value = results[metric]
                    if metric == 'MAPE':
                        print(f"  {model_name}: {value:.2f}%")
                    else:
                        print(f"  {model_name}: {value:.6f}")

if __name__ == "__main__":
    main()