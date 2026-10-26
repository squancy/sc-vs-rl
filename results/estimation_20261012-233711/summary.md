# Protocol 2 — estimation test
Each replication draws one observed history of 600 periods (50 years) from the true model. SC methods estimate parameters from it; RL methods train on windows of the same history (and, optionally, on a simulator of the fitted model). No method sees the true parameters. Gaps are to the true-parameter optimum for the same information set, evaluated on common test paths.

## gbm
True model `GBM(r=0.02, mu=0.08, sigma=0.2)`, information `returns`, reference `DP:discrete`, 3 histories.

| policy | mean gap (bp/yr) | sd across histories | median | bankrupt reps |
|---|---|---|---|---|
| SC:Rolling-shrink(60) | 24.8 | 28.1 | 8.7 | 0 |
| RL:DPO[fitted-sim] | 25.9 | 31.1 | 9.4 | 0 |
| SC:Merton-MLE | 27.7 | 33.3 | 10.8 | 0 |
| RL:DPO[history, τ-only] | 47.7 | 40.8 | 45.7 | 0 |
| SC:Rolling-Merton(120) | 101.6 | 0.0 | 101.6 | 0 |
| RL:DPO[history] | 118.4 | 67.8 | 154.1 | 0 |
| SC:Rolling-Merton(60) | 166.3 | 0.0 | 166.3 | 0 |

## ko_full
True model `KimOmberg(r=0.02, sigma=0.2, lam=0.2, Xbar=0.3, sigma_X=0.1265, rho=-0.9)`, information `full`, reference `DP:full-info`, 3 histories.

| policy | mean gap (bp/yr) | sd across histories | median | bankrupt reps |
|---|---|---|---|---|
| SC:KO-fitted(MLE) | 7.5 | 8.8 | 5.1 | 0 |
| RL:DPO[fitted-sim] | 15.5 | 18.3 | 9.5 | 0 |
| RL:DPO[history, τ-only] | 71.9 | 26.7 | 58.0 | 0 |
| SC:Merton-MLE | 71.2 | 21.5 | 60.4 | 0 |
| SC:Rolling-shrink(60) | 83.8 | 17.8 | 73.7 | 0 |
| SC:Rolling-Merton(120) | 199.1 | 0.0 | 199.1 | 0 |
| RL:DPO[history] | 154.6 | 99.6 | 200.5 | 0 |
| SC:Rolling-Merton(60) | 278.0 | 0.0 | 278.0 | 0 |

## ko_partial
True model `KimOmberg(r=0.02, sigma=0.2, lam=0.2, Xbar=0.3, sigma_X=0.1265, rho=-0.9)`, information `returns`, reference `DP:partial-info`, 3 histories.

| policy | mean gap (bp/yr) | sd across histories | median | bankrupt reps |
|---|---|---|---|---|
| Oracle:full-info (infeasible) | -27.9 | 0.0 | -27.9 | 0 |
| SC:KO-partial-fitted(Kalman MAP) | 19.0 | 12.4 | 13.0 | 0 |
| SC:Merton-MLE | 30.4 | 4.3 | 28.2 | 0 |
| RL:DPO[history, τ-only] | 47.2 | 28.1 | 31.9 | 0 |
| SC:Rolling-shrink(60) | 45.3 | 4.9 | 43.5 | 0 |
| RL:DPO[history] | 99.2 | 93.0 | 47.6 | 0 |
| SC:Rolling-Merton(120) | 175.7 | 0.0 | 175.7 | 0 |
| RL:DPO[fitted-sim] | 179.3 | 79.2 | 210.1 | 0 |
| SC:Rolling-Merton(60) | 253.6 | 0.0 | 253.6 | 0 |
| SC:KO-partial-fitted(Kalman MLE) | 298.1 | 156.6 | 341.1 | 0 |

## heston
True model `Heston(r=0.02, xi=1.5, kappa=2, vbar=0.04, sigma_v=0.4, rho=-0.7, n_sub=8)`, information `returns`, reference `SC:Kraft`, 3 histories.

| policy | mean gap (bp/yr) | sd across histories | median | bankrupt reps |
|---|---|---|---|---|
| SC:Kraft-fitted(RV moments) | 10.1 | 11.6 | 5.8 | 0 |
| RL:DPO[fitted-sim] | 9.9 | 9.2 | 7.1 | 0 |
| SC:Merton-MLE | 10.7 | 8.9 | 7.3 | 0 |
| SC:Rolling-shrink(60) | 55.9 | 39.7 | 35.2 | 0 |
| RL:DPO[history, τ-only] | 105.1 | 125.8 | 46.6 | 0 |
| RL:DPO[history] | 514.4 | 729.4 | 103.0 | 0 |
| SC:Rolling-Merton(120) | 128.1 | 0.0 | 128.1 | 0 |
| SC:VolTiming(EWMA, const premium) | 189.5 | 108.7 | 145.9 | 0 |
| SC:Rolling-Merton(60) | 204.6 | 0.0 | 204.6 | 0 |

## cv_returns
True model `ChackoViceira(r=0.02, mu_e=0.06, kappa=1, theta=30, sigma_y=3.162, rho=0.5, n_sub=21, y_floor=1)`, information `returns`, reference `Oracle:full-info (infeasible)`, 3 histories.

| policy | mean gap (bp/yr) | sd across histories | median | bankrupt reps |
|---|---|---|---|---|
| RL:DPO[fitted-sim] | 16.7 | 16.2 | 8.5 | 0 |
| SC:CV-fitted(RV moments) | 16.6 | 15.6 | 8.6 | 0 |
| RL:DPO[history, τ-only] | 26.8 | 13.8 | 19.1 | 0 |
| SC:Rolling-shrink(60) | 25.7 | 9.7 | 22.2 | 0 |
| SC:Merton-MLE | 31.8 | 13.8 | 36.4 | 0 |
| RL:DPO[history] | 112.1 | 16.9 | 107.6 | 0 |
| SC:Rolling-Merton(120) | 119.6 | 0.0 | 119.6 | 0 |
| SC:Rolling-Merton(60) | 178.9 | 0.0 | 178.9 | 0 |
