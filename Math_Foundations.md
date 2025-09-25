<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# 📘 Mathematical Foundations of Time Series Forecasting

This document summarizes the **mathematical principles** underlying the models implemented in this project.  

---

## 1. Classical Models  

### 1.1 ARIMA (AutoRegressive Integrated Moving Average)  
Model form:  

![ARIMA](https://latex.codecogs.com/svg.latex?\phi(L)(1-L)^d%20y_t%20=%20\theta(L)\varepsilon_t)

- $\phi(L)$: autoregressive (AR) polynomial  
- $(1-L)^d$: differencing operator (to achieve stationarity)  
- $\theta(L)$: moving average (MA) polynomial  
- $\varepsilon_t \sim \text{i.i.d.}(0,\sigma^2)$: white noise  

**Idea**: capture autocorrelation + non-stationary trend.  

---

### 1.2 GARCH (Generalized ARCH)  
![GARCH](https://latex.codecogs.com/svg.latex?y_t=\mu+\varepsilon_t,\;\varepsilon_t=\sigma_tz_t,\;z_t\sim%20N(0,1))  

![Var](https://latex.codecogs.com/svg.latex?\sigma_t^2=\alpha_0+\sum_{i=1}^p\alpha_i\varepsilon_{t-i}^2+\sum_{j=1}^q\beta_j\sigma_{t-j}^2)

**Idea**: conditional variance depends on past shocks and variances → volatility clustering.  

---

### 1.3 VAR (Vector Autoregression)  
![VAR](https://latex.codecogs.com/svg.latex?Y_t=A_1Y_{t-1}+A_2Y_{t-2}+\dots+A_pY_{t-p}+\varepsilon_t)

- $Y_t$: vector of $k$ time series  
- $A_i$: coefficient matrices  
- $\varepsilon_t \sim N(0, \Sigma)$: multivariate noise  

**Idea**: captures interdependence among multiple series.  

---

### 1.4 State-Space Models  
General form:  

![SSM](https://latex.codecogs.com/svg.latex?x_t=Fx_{t-1}+w_t,\;w_t\sim%20N(0,Q))  

![SSM2](https://latex.codecogs.com/svg.latex?y_t=Hx_t+v_t,\;v_t\sim%20N(0,R))

- $x_t$: hidden state  
- $y_t$: observation  
- $F, H$: transition and observation matrices  

**Idea**: separates latent dynamics (state) and noisy observation.  

---

## 2. Probabilistic Forecasting  

### 2.1 Kalman Filter  
- **Prediction**:  
![KF1](https://latex.codecogs.com/svg.latex?\hat{x}_{t|t-1}=F\hat{x}_{t-1|t-1},\;P_{t|t-1}=FP_{t-1|t-1}F^\top+Q)  

- **Update**:  
![KF2](https://latex.codecogs.com/svg.latex?K_t=P_{t|t-1}H^\top(HP_{t|t-1}H^\top+R)^{-1})  

![KF3](https://latex.codecogs.com/svg.latex?\hat{x}_{t|t}=\hat{x}_{t|t-1}+K_t(y_t-H\hat{x}_{t|t-1}))

---

### 2.2 Extended Kalman Filter  
Nonlinear system:  
![EKF](https://latex.codecogs.com/svg.latex?x_t=f(x_{t-1})+w_t,\;y_t=h(x_t)+v_t)  

---

### 2.3 Particle Filter  
Sequential importance sampling:  
![PF](https://latex.codecogs.com/svg.latex?p(x_{0:t}\mid%20y_{1:t})\approx\sum_{i=1}^Nw_t^{(i)}\delta(x_{0:t}-x_{0:t}^{(i)}))  

---

### 2.4 Bayesian Time Series Models  
Posterior inference:  
![Bayes](https://latex.codecogs.com/svg.latex?p(\theta\mid%20y_{1:T})\propto%20p(y_{1:T}\mid\theta)p(\theta))  

---

## 3. Deep Learning  

### 3.1 RNN / LSTM / GRU  
- **RNN**:  
![RNN](https://latex.codecogs.com/svg.latex?h_t=\sigma(W_hh_{t-1}+W_xx_t+b))  

- **LSTM**:  
![LSTM1](https://latex.codecogs.com/svg.latex?f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\;i_t=\sigma(W_i[h_{t-1},x_t]+b_i))  

![LSTM2](https://latex.codecogs.com/svg.latex?c_t=f_t\odot%20c_{t-1}+i_t\odot\tanh(W_c[h_{t-1},x_t]+b_c))  

![LSTM3](https://latex.codecogs.com/svg.latex?h_t=o_t\odot\tanh(c_t))  

---

### 3.2 Temporal Convolutional Networks (TCN)  
![TCN](https://latex.codecogs.com/svg.latex?h_t=\sum_{i=0}^kW_ix_{t-d\cdot%20i})  

---

### 3.3 Transformer  
Attention mechanism:  
![Attention](https://latex.codecogs.com/svg.latex?\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V)  

---

## 4. Evaluation Metrics  

### 4.1 Point Forecast  
- **RMSE**:  
![RMSE](https://latex.codecogs.com/svg.latex?\text{RMSE}=\sqrt{\frac{1}{T}\sum_{t=1}^T(y_t-\hat{y}_t)^2})  

- **MAE**:  
![MAE](https://latex.codecogs.com/svg.latex?\text{MAE}=\frac{1}{T}\sum_{t=1}^T|y_t-\hat{y}_t|)  

- **MAPE**:  
![MAPE](https://latex.codecogs.com/svg.latex?\text{MAPE}=\frac{100\%}{T}\sum_{t=1}^T\left|\frac{y_t-\hat{y}_t}{y_t}\right|)  

---

### 4.2 Probabilistic Metrics  
- **CRPS**:  
![CRPS](https://latex.codecogs.com/svg.latex?\text{CRPS}(F,y)=\int_{-\infty}^{\infty}(F(z)-\mathbf{1}\{y\leq%20z\})^2dz)  

- **Pinball Loss**:  
![Pinball](https://latex.codecogs.com/svg.latex?L_\tau(y,\hat{y}_\tau)=(y-\hat{y}_\tau)(\tau-\mathbf{1}_{\{y<\hat{y}_\tau\}}))  

---

### 4.3 Statistical Tests  
- **Diebold-Mariano Test**:  
Test null $H_0: E[d_t]=0$ where $d_t = L(e_{1t})-L(e_{2t})$.  
Checks if forecast accuracy of two models is significantly different.  

---




