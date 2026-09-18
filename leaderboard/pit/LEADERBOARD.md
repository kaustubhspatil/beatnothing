# Leaderboard

Universe: every S&P 500 member on each day, dead names included (point in time).

Universe bar: always long, equal weight, same universe, same costs. Costs: 10 bps per unit of turnover. Net Edge = net Sharpe minus the universe bar's net Sharpe on the same days, with a 95% paired stationary bootstrap interval (2,000 resamples, mean block 10 days). A contestant clears a bar only if the whole interval is above zero.
 Investable bar: RSP, the equal weight S&P 500 ETF, buy and hold, no cost charged because it is one position held throughout; it holds every index member by construction, dead ones included. The gap between the two bars is the survivorship and selection inflation of the contestant's universe.

## sealed 2022 2025  (2022-01-01 to 2025-12-31)

<table>
<tr><th>Contestant</th><th>Rule</th><th>Net Edge</th><th>95% CI</th><th>Clears bar</th><th>Edge vs RSP</th><th>95% CI</th><th>Clears RSP</th><th>Net Sharpe</th><th>Gross Sharpe</th><th>Sharpe at 20 bps</th><th>Max DD</th><th>Turnover/yr</th><th>Avg exposure</th><th>P&amp;L on $1M</th></tr>
<tr><td>momentum_12_1_long_short</td><td>long_short vs cash</td><td>+0.21</td><td>[-0.58, +1.06]</td><td>no</td><td>-0.23</td><td>[-1.68, +1.12]</td><td>no</td><td>+0.21</td><td>+0.27</td><td>+0.15</td><td>-16.7%</td><td>7.0x</td><td>0.0%</td><td>+70,526</td></tr>
<tr><td>momentum_12_1_long_top</td><td>long_top</td><td>+0.16</td><td>[-0.39, +0.75]</td><td>no</td><td>+0.19</td><td>[-0.42, +0.80]</td><td>no</td><td>+0.63</td><td>+0.66</td><td>+0.59</td><td>-23.7%</td><td>7.2x</td><td>100.0%</td><td>+561,740</td></tr>
<tr><td>low_vol_63d_long_top</td><td>long_top</td><td>+0.02</td><td>[-0.71, +0.71]</td><td>no</td><td>+0.05</td><td>[-0.68, +0.78]</td><td>no</td><td>+0.49</td><td>+0.55</td><td>+0.43</td><td>-14.3%</td><td>7.7x</td><td>100.0%</td><td>+232,140</td></tr>
<tr><td>costaware_net_lambda10bps_3seed_mean</td><td>long_flat</td><td>+0.00</td><td>[-0.04, +0.06]</td><td>no</td><td>+0.03</td><td>[-0.04, +0.11]</td><td>no</td><td>+0.47</td><td>+0.50</td><td>+0.44</td><td>-11.7%</td><td>3.2x</td><td>56.3%</td><td>+185,477</td></tr>
<tr><td>always_long</td><td>long_flat</td><td>+0.00</td><td>[+0.00, +0.00]</td><td>no</td><td>+0.03</td><td>[-0.01, +0.06]</td><td>no</td><td>+0.47</td><td>+0.47</td><td>+0.47</td><td>-20.8%</td><td>0.4x</td><td>100.0%</td><td>+298,376</td></tr>
<tr><td>ffnn_128_64_32</td><td>long_flat</td><td>-0.02</td><td>[-0.03, -0.01]</td><td>no</td><td>+0.01</td><td>[-0.03, +0.05]</td><td>no</td><td>+0.45</td><td>+0.47</td><td>+0.42</td><td>-20.9%</td><td>4.3x</td><td>100.0%</td><td>+282,761</td></tr>
<tr><td>lightgbm_mse</td><td>long_flat</td><td>-0.11</td><td>[-0.39, +0.11]</td><td>no</td><td>-0.08</td><td>[-0.37, +0.13]</td><td>no</td><td>+0.36</td><td>+0.40</td><td>+0.31</td><td>-22.4%</td><td>7.4x</td><td>98.4%</td><td>+195,297</td></tr>
<tr><td>lstm_60d</td><td>long_flat</td><td>-0.25</td><td>[-0.51, -0.03]</td><td>no</td><td>-0.22</td><td>[-0.46, +0.00]</td><td>no</td><td>+0.21</td><td>+0.52</td><td>-0.09</td><td>-22.4%</td><td>50.5x</td><td>100.0%</td><td>+91,256</td></tr>
<tr><td>low_vol_63d_long_short</td><td>long_short vs cash</td><td>-0.27</td><td>[-1.21, +0.60]</td><td>no</td><td>-0.71</td><td>[-2.35, +0.92]</td><td>no</td><td>-0.27</td><td>-0.22</td><td>-0.32</td><td>-27.6%</td><td>6.7x</td><td>-0.0%</td><td>-168,504</td></tr>
<tr><td>costaware_net_lambda0bps_3seed_mean</td><td>long_flat</td><td>-0.33</td><td>[-0.87, +0.20]</td><td>no</td><td>-0.30</td><td>[-0.84, +0.24]</td><td>no</td><td>+0.14</td><td>+0.49</td><td>-0.20</td><td>-5.7%</td><td>12.1x</td><td>11.9%</td><td>+17,358</td></tr>
<tr><td>reversal_1m_long_top</td><td>long_top</td><td>-0.38</td><td>[-0.82, +0.01]</td><td>no</td><td>-0.35</td><td>[-0.79, +0.08]</td><td>no</td><td>+0.09</td><td>+0.17</td><td>+0.00</td><td>-27.4%</td><td>20.9x</td><td>100.0%</td><td>-35,323</td></tr>
<tr><td>reversal_1m_long_short</td><td>long_short vs cash</td><td>-0.47</td><td>[-1.37, +0.40]</td><td>no</td><td>-0.91</td><td>[-1.98, +0.16]</td><td>no</td><td>-0.47</td><td>-0.25</td><td>-0.68</td><td>-20.6%</td><td>20.8x</td><td>-0.0%</td><td>-179,846</td></tr>
<tr><td>linear_incumbent</td><td>long_flat</td><td>-0.80</td><td>[-1.15, -0.51]</td><td>no</td><td>-0.77</td><td>[-1.10, -0.48]</td><td>no</td><td>-0.34</td><td>+0.47</td><td>-1.14</td><td>-38.0%</td><td>150.9x</td><td>100.0%</td><td>-274,198</td></tr>
<tr><td>cnn1d_60d</td><td>long_flat</td><td>-1.03</td><td>[-1.40, -0.69]</td><td>no</td><td>-1.00</td><td>[-1.42, -0.67]</td><td>no</td><td>-0.56</td><td>+0.53</td><td>-1.66</td><td>-52.0%</td><td>234.8x</td><td>100.0%</td><td>-434,622</td></tr>
</table>

Universe bar net Sharpe +0.47 against RSP +0.44: the universe bar's edge over the investable index is +0.03 [-0.01, +0.06]. That is how much choosing the universe with hindsight was worth on this window.

## post cutoff 2026  (2026-01-01 to 2026-12-31)

<table>
<tr><th>Contestant</th><th>Rule</th><th>Net Edge</th><th>95% CI</th><th>Clears bar</th><th>Edge vs RSP</th><th>95% CI</th><th>Clears RSP</th><th>Net Sharpe</th><th>Gross Sharpe</th><th>Sharpe at 20 bps</th><th>Max DD</th><th>Turnover/yr</th><th>Avg exposure</th><th>P&amp;L on $1M</th></tr>
<tr><td>momentum_12_1_long_short</td><td>long_short vs cash</td><td>+0.57</td><td>[-1.10, +2.67]</td><td>no</td><td>-0.68</td><td>[-3.47, +2.11]</td><td>no</td><td>+0.57</td><td>+0.60</td><td>+0.53</td><td>-19.9%</td><td>7.1x</td><td>0.0%</td><td>+73,382</td></tr>
<tr><td>always_long</td><td>long_flat</td><td>+0.00</td><td>[+0.00, +0.00]</td><td>no</td><td>-0.01</td><td>[-0.32, +0.32]</td><td>no</td><td>+1.23</td><td>+1.25</td><td>+1.22</td><td>-7.7%</td><td>1.6x</td><td>100.0%</td><td>+99,898</td></tr>
<tr><td>lightgbm_mse</td><td>long_flat</td><td>+0.00</td><td>[+0.00, +0.00]</td><td>no</td><td>-0.01</td><td>[-0.32, +0.32]</td><td>no</td><td>+1.23</td><td>+1.25</td><td>+1.22</td><td>-7.7%</td><td>1.6x</td><td>100.0%</td><td>+99,898</td></tr>
<tr><td>costaware_net_lambda10bps_3seed_mean</td><td>long_flat</td><td>-0.02</td><td>[-0.11, +0.08]</td><td>no</td><td>-0.03</td><td>[-0.38, +0.37]</td><td>no</td><td>+1.21</td><td>+1.26</td><td>+1.16</td><td>-4.6%</td><td>3.5x</td><td>55.9%</td><td>+55,953</td></tr>
<tr><td>ffnn_128_64_32</td><td>long_flat</td><td>-0.02</td><td>[-0.07, +0.03]</td><td>no</td><td>-0.04</td><td>[-0.33, +0.29]</td><td>no</td><td>+1.21</td><td>+1.27</td><td>+1.15</td><td>-7.8%</td><td>6.8x</td><td>100.0%</td><td>+97,955</td></tr>
<tr><td>cnn1d_60d</td><td>long_flat</td><td>-0.08</td><td>[-1.36, +1.42]</td><td>no</td><td>-0.09</td><td>[-1.35, +1.29]</td><td>no</td><td>+1.16</td><td>+2.18</td><td>+0.13</td><td>-14.2%</td><td>206.0x</td><td>100.0%</td><td>+159,144</td></tr>
<tr><td>momentum_12_1_long_top</td><td>long_top</td><td>-0.09</td><td>[-2.24, +2.13]</td><td>no</td><td>-0.10</td><td>[-2.34, +1.88]</td><td>no</td><td>+1.15</td><td>+1.17</td><td>+1.13</td><td>-22.9%</td><td>6.7x</td><td>100.0%</td><td>+252,969</td></tr>
<tr><td>costaware_net_lambda0bps_3seed_mean</td><td>long_flat</td><td>-0.25</td><td>[-0.89, +0.43]</td><td>no</td><td>-0.26</td><td>[-1.12, +0.65]</td><td>no</td><td>+0.98</td><td>+1.72</td><td>+0.25</td><td>-1.0%</td><td>10.3x</td><td>9.9%</td><td>+9,578</td></tr>
<tr><td>linear_incumbent</td><td>long_flat</td><td>-0.54</td><td>[-1.51, +0.39]</td><td>no</td><td>-0.55</td><td>[-1.58, +0.43]</td><td>no</td><td>+0.69</td><td>+1.69</td><td>-0.30</td><td>-8.6%</td><td>139.7x</td><td>100.0%</td><td>+62,491</td></tr>
<tr><td>lstm_60d</td><td>long_flat</td><td>-0.74</td><td>[-2.39, +0.41]</td><td>no</td><td>-0.75</td><td>[-2.47, +0.47]</td><td>no</td><td>+0.49</td><td>+0.89</td><td>+0.10</td><td>-8.8%</td><td>47.2x</td><td>100.0%</td><td>+36,514</td></tr>
<tr><td>reversal_1m_long_top</td><td>long_top</td><td>-0.95</td><td>[-3.17, +1.26]</td><td>no</td><td>-0.96</td><td>[-3.35, +1.37]</td><td>no</td><td>+0.28</td><td>+0.39</td><td>+0.17</td><td>-13.0%</td><td>21.1x</td><td>100.0%</td><td>+25,395</td></tr>
<tr><td>low_vol_63d_long_top</td><td>long_top</td><td>-1.06</td><td>[-3.45, +1.27]</td><td>no</td><td>-1.07</td><td>[-3.46, +1.29]</td><td>no</td><td>+0.18</td><td>+0.24</td><td>+0.11</td><td>-7.7%</td><td>6.9x</td><td>100.0%</td><td>+9,302</td></tr>
<tr><td>low_vol_63d_long_short</td><td>long_short vs cash</td><td>-1.32</td><td>[-3.53, +0.51]</td><td>no</td><td>-2.57</td><td>[-5.84, +0.37]</td><td>no</td><td>-1.32</td><td>-1.28</td><td>-1.36</td><td>-19.0%</td><td>7.0x</td><td>0.0%</td><td>-162,525</td></tr>
<tr><td>reversal_1m_long_short</td><td>long_short vs cash</td><td>-1.73</td><td>[-4.36, +0.93]</td><td>no</td><td>-2.98</td><td>[-6.47, +0.39]</td><td>no</td><td>-1.73</td><td>-1.57</td><td>-1.89</td><td>-21.5%</td><td>20.8x</td><td>0.0%</td><td>-149,537</td></tr>
</table>

Universe bar net Sharpe +1.23 against RSP +1.25: the universe bar's edge over the investable index is -0.01 [-0.32, +0.32]. That is how much choosing the universe with hindsight was worth on this window.

## Contestants

* **always_long** (predictions, training cutoff 2018-12-31, registered 2026-09-18): The universe bar: long every member every day.
* **cnn1d_60d** (predictions, training cutoff 2018-12-31, registered 2026-09-18): Three block 1D CNN over 60 day windows, frozen 2026 08 20, scored on every member.
* **costaware_net_lambda0bps_3seed_mean** (weights, training cutoff 2018-12-31, registered 2026-09-18): Cost aware network, 0 bps turnover term, mean of seeds 42 43 44, weights = sigmoid / N over the members present that day.
* **costaware_net_lambda10bps_3seed_mean** (weights, training cutoff 2018-12-31, registered 2026-09-18): Cost aware network, 10 bps turnover term, mean of seeds 42 43 44, weights = sigmoid / N over the members present that day.
* **ffnn_128_64_32** (predictions, training cutoff 2018-12-31, registered 2026-09-18): Feedforward 128 64 32, MSE, frozen 2026 08 20, scored on every member.
* **lightgbm_mse** (predictions, training cutoff 2018-12-31, registered 2026-09-18): LightGBM, MSE objective, early stopped after three trees, scored on every member.
* **linear_incumbent** (predictions, training cutoff 2018-12-31, registered 2026-09-18): OLS on the 17 trailing features, frozen 2026 08 20, scored on every member.
* **low_vol_63d_long_short** (predictions, training cutoff none (a formula), registered 2026-09-18): Low volatility: minus the trailing 63 day realised volatility. Monthly rebalanced. Long the top decile, short the bottom decile, dollar neutral, 50 bps a year borrow.
* **low_vol_63d_long_top** (predictions, training cutoff none (a formula), registered 2026-09-18): Low volatility: minus the trailing 63 day realised volatility. Monthly rebalanced. Long only the top decile, equal weight; judged against the universe bar.
* **lstm_60d** (predictions, training cutoff 2018-12-31, registered 2026-09-18): Two layer LSTM over 60 day windows, frozen 2026 08 20, scored on every member.
* **momentum_12_1_long_short** (predictions, training cutoff none (a formula), registered 2026-09-18): Twelve month momentum skipping the most recent month (Jegadeesh and Titman). Monthly rebalanced. Long the top decile, short the bottom decile, dollar neutral, 50 bps a year borrow.
* **momentum_12_1_long_top** (predictions, training cutoff none (a formula), registered 2026-09-18): Twelve month momentum skipping the most recent month (Jegadeesh and Titman). Monthly rebalanced. Long only the top decile, equal weight; judged against the universe bar.
* **reversal_1m_long_short** (predictions, training cutoff none (a formula), registered 2026-09-18): One month short term reversal: minus the trailing 21 day return. Monthly rebalanced. Long the top decile, short the bottom decile, dollar neutral, 50 bps a year borrow.
* **reversal_1m_long_top** (predictions, training cutoff none (a formula), registered 2026-09-18): One month short term reversal: minus the trailing 21 day return. Monthly rebalanced. Long only the top decile, equal weight; judged against the universe bar.
