# Protocol 1 — oracle test
gamma = 5.0, horizon = 10.0 years, dt = 0.0833, pre-sample = 60 periods, bounds = [-1.0, 2.0], test paths = 20000 (common random numbers).
Gap = (log CE_ref − log CE_policy)/T in basis points per year (positive = worse than the reference), ± delta-method standard error on paired paths.

## gbm
Model: `GBM(r=0.02, mu=0.08, sigma=0.2)`, information: `returns`, reference: `DP:discrete`.

| policy | feasible | CE rate | gap to ref (bp/yr) | π RMSE vs ref | mean π | turnover | P5 W_T | bankrupt |
|---|---|---|---|---|---|---|---|---|
| SC:Merton | yes | 2.895% | 0.0 ± 0.0 | 0.0001 | 0.300 | 0.0000 | 1.050 | 0 |
| DP:discrete (ref) | yes | 2.895% | -0.0 ± 0.0 | 0.0000 | 0.300 | 0.0000 | 1.050 | 0 |
| Best-constant | yes | 2.895% | 0.1 ± 0.0 | 0.0066 | 0.306 | 0.0000 | 1.046 | 0 |
| RL:DPO | yes | 2.892% | 0.4 ± 0.1 | 0.0177 | 0.296 | 0.0057 | 1.049 | 0 |
| RL:PPO | yes | 2.888% | 0.7 ± 0.2 | 0.0258 | 0.294 | 0.0083 | 1.051 | 0 |
| RL:TD3 | yes | 2.859% | 3.7 ± 0.4 | 0.0612 | 0.256 | 0.0143 | 1.073 | 0 |

## jump
Model: `MertonJump(r=0.02, mu=0.08, sigma=0.1446, lam=0.5, muJ=-0.2, sigJ=0.1)`, information: `returns`, reference: `DP:discrete`.

| policy | feasible | CE rate | gap to ref (bp/yr) | π RMSE vs ref | mean π | turnover | P5 W_T | bankrupt |
|---|---|---|---|---|---|---|---|---|
| SC:JumpFOC | yes | 2.853% | -0.0 ± 0.0 | 0.0003 | 0.271 | 0.0000 | 1.060 | 0 |
| SC:Merton(total var) | yes | 2.844% | 0.9 ± 0.3 | 0.0296 | 0.300 | 0.0000 | 1.041 | 0 |
| DP:discrete (ref) | yes | 2.853% | -0.0 ± 0.0 | 0.0000 | 0.270 | 0.0000 | 1.060 | 0 |
| Best-constant | yes | 2.853% | -0.0 ± 0.0 | 0.0039 | 0.274 | 0.0000 | 1.057 | 0 |
| RL:DPO | yes | 2.848% | 0.5 ± 0.1 | 0.0190 | 0.278 | 0.0037 | 1.050 | 0 |
| RL:PPO | yes | 2.841% | 1.2 ± 0.3 | 0.0274 | 0.256 | 0.0079 | 1.065 | 0 |
| RL:TD3 | yes | 2.575% | 27.7 ± 1.0 | 0.1477 | 0.149 | 0.0337 | 1.107 | 0 |

## heston
Model: `Heston(r=0.02, xi=1.5, kappa=2, vbar=0.04, sigma_v=0.4, rho=-0.7, n_sub=8)`, information: `full`, reference: `SC:Kraft`.

| policy | feasible | CE rate | gap to ref (bp/yr) | π RMSE vs ref | mean π | turnover | P5 W_T | bankrupt |
|---|---|---|---|---|---|---|---|---|
| SC:Kraft (ref) | yes | 3.040% | -0.0 ± 0.0 | 0.0000 | 0.328 | 0.0002 | 1.072 | 0 |
| SC:Myopic | yes | 3.033% | 0.7 ± 0.4 | 0.0285 | 0.300 | 0.0000 | 1.088 | 0 |
| Best-constant | yes | 3.039% | 0.1 ± 0.2 | 0.0107 | 0.338 | 0.0000 | 1.068 | 0 |
| RL:DPO | yes | 3.040% | 0.1 ± 0.1 | 0.0206 | 0.315 | 0.0085 | 1.074 | 0 |
| RL:PPO | yes | 3.019% | 2.2 ± 0.7 | 0.0546 | 0.306 | 0.0299 | 1.064 | 0 |
| RL:TD3 | yes | 2.860% | 18.1 ± 1.3 | 0.0711 | 0.289 | 0.0279 | 1.063 | 0 |

## cv_full
Model: `ChackoViceira(r=0.02, mu_e=0.06, kappa=1, theta=30, sigma_y=3.162, rho=0.5, n_sub=21, y_floor=1)`, information: `full`, reference: `SC:ChackoViceira`.

| policy | feasible | CE rate | gap to ref (bp/yr) | π RMSE vs ref | mean π | turnover | P5 W_T | bankrupt |
|---|---|---|---|---|---|---|---|---|
| SC:ChackoViceira (ref) | yes | 2.995% | -0.0 ± 0.0 | 0.0000 | 0.349 | 0.0446 | 1.049 | 0 |
| SC:CV-myopic | yes | 2.993% | 0.2 ± 0.1 | 0.0127 | 0.360 | 0.0461 | 1.043 | 0 |
| Best-constant | yes | 2.837% | 15.8 ± 0.8 | 0.1653 | 0.265 | 0.0000 | 1.058 | 0 |
| RL:DPO | yes | 2.993% | 0.2 ± 0.1 | 0.0369 | 0.340 | 0.0391 | 1.051 | 0 |
| RL:PPO | yes | 2.983% | 1.2 ± 0.2 | 0.0664 | 0.328 | 0.0334 | 1.055 | 0 |
| RL:TD3 | yes | 2.893% | 10.2 ± 0.6 | 0.0860 | 0.290 | 0.0595 | 1.073 | 0 |

## cv_returns
Model: `ChackoViceira(r=0.02, mu_e=0.06, kappa=1, theta=30, sigma_y=3.162, rho=0.5, n_sub=21, y_floor=1)`, information: `returns`, reference: `Oracle:full-info (infeasible)`.

| policy | feasible | CE rate | gap to ref (bp/yr) | π RMSE vs ref | mean π | turnover | P5 W_T | bankrupt |
|---|---|---|---|---|---|---|---|---|
| SC:CV-plugin(rv, hl=2) | yes | 2.956% | 4.2 ± 0.4 | 0.0818 | 0.342 | 0.0309 | 1.044 | 0 |
| SC:CV-unconditional | yes | 2.822% | 17.6 ± 1.0 | 0.1507 | 0.300 | 0.0000 | 1.032 | 0 |
| Oracle:full-info (infeasible) (ref) | no | 2.998% | -0.0 ± 0.0 | 0.0000 | 0.349 | 0.0445 | 1.050 | 0 |
| Best-constant | yes | 2.836% | 16.2 ± 0.8 | 0.1625 | 0.271 | 0.0000 | 1.051 | 0 |
| RL:DPO | yes | 2.947% | 5.1 ± 0.4 | 0.0964 | 0.324 | 0.0259 | 1.046 | 0 |
| RL:PPO | yes | 2.931% | 6.7 ± 0.5 | 0.1078 | 0.295 | 0.0232 | 1.063 | 0 |
| RL:TD3 | yes | 2.556% | 44.2 ± 1.2 | 0.2168 | 0.197 | 0.0484 | 1.066 | 0 |

## ko_full
Model: `KimOmberg(r=0.02, sigma=0.2, lam=0.2, Xbar=0.3, sigma_X=0.1265, rho=-0.9)`, information: `full`, reference: `DP:full-info`.

| policy | feasible | CE rate | gap to ref (bp/yr) | π RMSE vs ref | mean π | turnover | P5 W_T | bankrupt |
|---|---|---|---|---|---|---|---|---|
| SC:KimOmberg | yes | 3.915% | -0.0 ± 0.0 | 0.0009 | 0.414 | 0.0361 | 1.163 | 0 |
| SC:Myopic | yes | 3.724% | 19.1 ± 1.5 | 0.1381 | 0.300 | 0.0290 | 1.185 | 0 |
| DP:full-info (ref) | yes | 3.915% | -0.0 ± 0.0 | 0.0000 | 0.414 | 0.0360 | 1.163 | 0 |
| Best-constant | yes | 3.327% | 58.8 ± 2.0 | 0.2568 | 0.444 | 0.0000 | 1.070 | 0 |
| RL:DPO | yes | 3.882% | 3.3 ± 0.5 | 0.0379 | 0.424 | 0.0361 | 1.148 | 0 |
| RL:PPO | yes | 3.679% | 23.6 ± 1.7 | 0.1402 | 0.325 | 0.0231 | 1.155 | 0 |
| RL:TD3 | yes | 3.809% | 10.5 ± 0.9 | 0.0843 | 0.395 | 0.0393 | 1.125 | 0 |

## ko_partial
Model: `KimOmberg(r=0.02, sigma=0.2, lam=0.2, Xbar=0.3, sigma_X=0.1265, rho=-0.9)`, information: `returns`, reference: `DP:partial-info`.

| policy | feasible | CE rate | gap to ref (bp/yr) | π RMSE vs ref | mean π | turnover | P5 W_T | bankrupt |
|---|---|---|---|---|---|---|---|---|
| SC:KO-partial | yes | 3.643% | -0.0 ± 0.0 | 0.0007 | 0.390 | 0.0258 | 1.131 | 0 |
| SC:KO-cert-equiv | yes | 3.639% | 0.3 ± 0.6 | 0.0305 | 0.416 | 0.0269 | 1.122 | 0 |
| SC:Myopic-filtered | yes | 3.521% | 12.1 ± 1.4 | 0.1031 | 0.301 | 0.0216 | 1.157 | 0 |
| DP:partial-info (ref) | yes | 3.643% | -0.0 ± 0.0 | 0.0000 | 0.390 | 0.0257 | 1.131 | 0 |
| Oracle:full-info (infeasible) | no | 3.862% | -21.9 ± 5.9 | 0.1772 | 0.414 | 0.0361 | 1.165 | 0 |
| Best-constant | yes | 3.352% | 29.1 ± 1.8 | 0.1917 | 0.469 | 0.0000 | 1.067 | 0 |
| RL:DPO | yes | 3.622% | 2.0 ± 0.7 | 0.0362 | 0.395 | 0.0285 | 1.117 | 0 |
| RL:PPO | yes | 3.518% | 12.4 ± 1.7 | 0.0865 | 0.341 | 0.0171 | 1.132 | 0 |
| RL:TD3 | yes | 2.830% | 81.3 ± 2.6 | 0.2849 | 0.130 | 0.0264 | 1.096 | 0 |
