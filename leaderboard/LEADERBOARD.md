# Leaderboard

Bar: always long, equal weight, same universe, same costs. Costs: 10 bps per unit of turnover. Net Edge = net Sharpe minus the bar's net Sharpe on the same days, with a 95% paired stationary bootstrap interval (2,000 resamples, mean block 10 days). A contestant clears the bar only if the whole interval is above zero.

## sealed 2022 2025  (2022-01-01 to 2025-12-31)

<table>
<tr><th>Contestant</th><th>Net Edge</th><th>95% CI</th><th>Clears bar</th><th>Net Sharpe</th><th>Gross Sharpe</th><th>Sharpe at 20 bps</th><th>Max DD</th><th>Turnover/yr</th><th>Avg exposure</th><th>P&amp;L on $1M</th></tr>
<tr><td>always_long</td><td>+0.00</td><td>[+0.00, +0.00]</td><td>no</td><td>+0.88</td><td>+0.88</td><td>+0.88</td><td>-20.9%</td><td>0.3x</td><td>100.0%</td><td>+663,322</td></tr>
<tr><td>ffnn_128_64_32</td><td>-0.03</td><td>[-0.06, -0.01]</td><td>no</td><td>+0.85</td><td>+0.88</td><td>+0.82</td><td>-21.3%</td><td>4.3x</td><td>100.0%</td><td>+631,816</td></tr>
<tr><td>costaware_net_lambda10bps_3seed_mean</td><td>-0.04</td><td>[-0.10, +0.01]</td><td>no</td><td>+0.83</td><td>+0.87</td><td>+0.80</td><td>-12.0%</td><td>2.9x</td><td>56.0%</td><td>+345,138</td></tr>
<tr><td>lightgbm_mse</td><td>-0.14</td><td>[-0.45, +0.09]</td><td>no</td><td>+0.74</td><td>+0.79</td><td>+0.69</td><td>-22.5%</td><td>7.2x</td><td>98.4%</td><td>+495,796</td></tr>
<tr><td>lstm_60d</td><td>-0.53</td><td>[-1.07, -0.10]</td><td>no</td><td>+0.35</td><td>+0.62</td><td>+0.08</td><td>-20.5%</td><td>42.7x</td><td>98.5%</td><td>+189,470</td></tr>
<tr><td>costaware_net_lambda0bps_3seed_mean</td><td>-0.70</td><td>[-1.33, -0.09]</td><td>no</td><td>+0.18</td><td>+0.53</td><td>-0.17</td><td>-6.4%</td><td>11.5x</td><td>11.6%</td><td>+21,510</td></tr>
<tr><td>cnn1d_60d</td><td>-0.93</td><td>[-1.62, -0.32]</td><td>no</td><td>-0.05</td><td>+1.08</td><td>-1.18</td><td>-34.9%</td><td>265.2x</td><td>98.9%</td><td>-145,881</td></tr>
<tr><td>linear_incumbent</td><td>-1.17</td><td>[-1.85, -0.64]</td><td>no</td><td>-0.30</td><td>+0.51</td><td>-1.10</td><td>-41.2%</td><td>159.2x</td><td>99.0%</td><td>-267,985</td></tr>
<tr><td>kronos_small_zero_shot</td><td>-1.67</td><td>[-2.13, -1.28]</td><td>no</td><td>-0.79</td><td>+0.52</td><td>-2.10</td><td>-53.0%</td><td>240.6x</td><td>100.0%</td><td>-475,324</td></tr>
</table>

## post cutoff 2026  (2026-01-01 to 2026-12-31)

<table>
<tr><th>Contestant</th><th>Net Edge</th><th>95% CI</th><th>Clears bar</th><th>Net Sharpe</th><th>Gross Sharpe</th><th>Sharpe at 20 bps</th><th>Max DD</th><th>Turnover/yr</th><th>Avg exposure</th><th>P&amp;L on $1M</th></tr>
<tr><td>costaware_net_lambda0bps_3seed_mean</td><td>+0.18</td><td>[-0.58, +0.97]</td><td>no</td><td>+1.62</td><td>+2.47</td><td>+0.78</td><td>-0.7%</td><td>9.7x</td><td>9.4%</td><td>+13,051</td></tr>
<tr><td>costaware_net_lambda10bps_3seed_mean</td><td>+0.00</td><td>[-0.07, +0.08]</td><td>no</td><td>+1.45</td><td>+1.50</td><td>+1.39</td><td>-3.6%</td><td>3.3x</td><td>55.5%</td><td>+58,151</td></tr>
<tr><td>always_long</td><td>+0.00</td><td>[+0.00, +0.00]</td><td>no</td><td>+1.44</td><td>+1.46</td><td>+1.43</td><td>-6.1%</td><td>1.4x</td><td>100.0%</td><td>+103,176</td></tr>
<tr><td>lightgbm_mse</td><td>+0.00</td><td>[+0.00, +0.00]</td><td>no</td><td>+1.44</td><td>+1.46</td><td>+1.43</td><td>-6.1%</td><td>1.4x</td><td>100.0%</td><td>+103,176</td></tr>
<tr><td>ffnn_128_64_32</td><td>-0.08</td><td>[-0.17, -0.01]</td><td>no</td><td>+1.36</td><td>+1.43</td><td>+1.29</td><td>-6.1%</td><td>7.6x</td><td>100.0%</td><td>+97,155</td></tr>
<tr><td>cnn1d_60d</td><td>-1.04</td><td>[-2.76, +0.76]</td><td>no</td><td>+0.40</td><td>+2.12</td><td>-1.31</td><td>-12.0%</td><td>323.1x</td><td>96.6%</td><td>+41,465</td></tr>
<tr><td>linear_incumbent</td><td>-1.33</td><td>[-3.21, +0.20]</td><td>no</td><td>+0.12</td><td>+1.35</td><td>-1.13</td><td>-9.1%</td><td>164.8x</td><td>100.0%</td><td>+4,546</td></tr>
<tr><td>kronos_small_zero_shot</td><td>-1.60</td><td>[-3.14, -0.12]</td><td>no</td><td>-0.19</td><td>+1.71</td><td>-2.08</td><td>-14.3%</td><td>244.7x</td><td>100.0%</td><td>-22,364</td></tr>
<tr><td>lstm_60d</td><td>-1.65</td><td>[-3.69, -0.07]</td><td>no</td><td>-0.20</td><td>+0.20</td><td>-0.60</td><td>-9.3%</td><td>42.8x</td><td>100.0%</td><td>-19,077</td></tr>
</table>

## Contestants

* **always_long** (predictions, training cutoff 2018-12-31, registered 2026-09-17): The do nothing bar itself: long every name every day. Anything that cannot beat this has no edge.
* **cnn1d_60d** (predictions, training cutoff 2018-12-31, registered 2026-09-17): Three block 1D CNN over 60 day windows. Frozen 2026 08 20.
* **costaware_net_lambda0bps_3seed_mean** (weights, training cutoff 2018-12-31, registered 2026-09-17): Cost aware end to end network: outputs weights, trained to maximise net Sharpe over 126 day windows with a turnover penalty of 0 bps. Mean of three seeds (42, 43, 44). Trained 2026 09 17.
* **costaware_net_lambda10bps_3seed_mean** (weights, training cutoff 2018-12-31, registered 2026-09-17): Cost aware end to end network: outputs weights, trained to maximise net Sharpe over 126 day windows with a turnover penalty of 10 bps. Mean of three seeds (42, 43, 44). Trained 2026 09 17.
* **ffnn_128_64_32** (predictions, training cutoff 2018-12-31, registered 2026-09-17): Feedforward network 128 64 32 trained on MSE with early stopping, the capstone's selected model. Frozen 2026 08 20.
* **kronos_small_zero_shot** (predictions, training cutoff pretrained by its authors (Kronos, AAAI 2026); never trained on this data, registered 2026-09-17): Kronos small (24.7M parameters, MIT) zero shot: reads the trailing 400 daily OHLCV bars of each stock and generates the next bar; the signal is the predicted close over the last close minus one, averaged over three samples (T=1.0, top_p=0.9). Weights NeoQuasar/Kronos-small and NeoQuasar/Kronos-Tokenizer-base from Hugging Face; script in contestants/kronos_zero_shot.py.
* **lightgbm_mse** (predictions, training cutoff 2018-12-31, registered 2026-09-17): LightGBM regressor on the same 17 features, MSE objective, early stopped on 2019 to 2021 validation (stopped after 3 trees). Trained 2026 09 17 on data to 2018.
* **linear_incumbent** (predictions, training cutoff 2018-12-31, registered 2026-09-17): Ordinary least squares on the 17 trailing features, the stand in for a rules based desk signal. Frozen 2026 08 20.
* **lstm_60d** (predictions, training cutoff 2018-12-31, registered 2026-09-17): Two layer LSTM over 60 day windows of the same features. Frozen 2026 08 20.
