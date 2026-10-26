# Protocol 3 — misspecification test
Information set: returns only. Rows are policies trained/calibrated on one model (SC[x] is the optimal returns-only SC policy of an investor who believes model x with its true parameters; RL[x] is trained on simulated paths of x; RL[mix] on an equal mixture of the five base models). Columns are the true evaluation model. Entries: CE gap (bp/yr) to the correctly specified SC policy of the column model.

| policy | gbm | jump | heston | cv | ko | gbm_highvol | ko_fast |
|---|---|---|---|---|---|---|---|
| SC[gbm] | -0.0 | 1.4 | 1.0 | 10.3 | 42.8 | 59.2 | 45.3 |
| SC[jump] | 0.7 | -0.0 | 3.8 | 9.7 | 49.1 | 40.0 | 51.3 |
| SC[heston] | 1.0 | 4.9 | -0.0 | 12.9 | 37.8 | 81.1 | 40.6 |
| SC[cv] | 40.9 | 207.4 | 65.1 | -0.0 | 67.7 | 18.1 | 69.7 |
| SC[ko] | 173.4 | 512.3 | 531.5 | 521.1 | -0.0 | 2196.2 | 19.3 |
| RL:DPO[gbm] | 0.4 | 2.0 | 6.5 | 9.1 | 41.3 | 7.8 | 43.8 |
| RL:DPO[jump] | 0.9 | 0.4 | 3.7 | 10.2 | 49.2 | 39.4 | 51.5 |
| RL:DPO[heston] | 1.9 | 5.2 | 0.6 | 14.4 | 36.0 | 64.8 | 37.9 |
| RL:DPO[cv] | 4.2 | 9.4 | 18.9 | 4.5 | 53.7 | 14.7 | 62.9 |
| RL:DPO[ko] | 98.1 | 95.0 | 61.7 | 148.3 | 3.3 | 303.1 | 21.3 |
| RL:DPO[mix] | 1.3 | 2.0 | 3.2 | 9.5 | 36.2 | 19.5 | 41.1 |

## Proportional transaction costs at evaluation time

### gbm@0.005

| policy | feasible | CE rate | gap to ref (bp/yr) | π RMSE vs ref | mean π | turnover | P5 W_T | bankrupt |
|---|---|---|---|---|---|---|---|---|
| SC[gbm] (ignores costs) | yes | 2.818% | 4.3 ± 0.2 | 0.0266 | 0.300 | 0.0000 | 1.045 | 0 |
| SC[gbm]+band (ref) | yes | 2.861% | -0.0 ± 0.0 | 0.0000 | 0.306 | 0.0081 | 1.049 | 0 |
| RL:DPO[gbm] (trained w/o costs) | yes | 2.789% | 7.2 ± 0.2 | 0.0367 | 0.300 | 0.0076 | 1.040 | 0 |
| RL:DPO[gbm, cost=0.005] | yes | 2.854% | 0.7 ± 0.1 | 0.0179 | 0.298 | 0.0084 | 1.050 | 0 |

### ko@0.005

| policy | feasible | CE rate | gap to ref (bp/yr) | π RMSE vs ref | mean π | turnover | P5 W_T | bankrupt |
|---|---|---|---|---|---|---|---|---|
| SC[ko] (ignores costs) | yes | 3.382% | 10.9 ± 0.2 | 0.0321 | 0.389 | 0.0257 | 1.099 | 0 |
| SC[ko]+band (ref) | yes | 3.491% | -0.0 ± 0.0 | 0.0000 | 0.391 | 0.0138 | 1.110 | 0 |
| RL:DPO[ko] (trained w/o costs) | yes | 3.370% | 12.1 ± 0.5 | 0.0477 | 0.396 | 0.0242 | 1.084 | 0 |
| RL:DPO[ko, cost=0.005] | yes | 3.457% | 3.4 ± 0.4 | 0.0497 | 0.409 | 0.0113 | 1.089 | 0 |
