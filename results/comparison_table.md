### 1. Accuracy

```
model      CV Top-1  Student Top-1  Student Top-1 (true road was a candidate)
   lr 0.881 ± 0.013          0.866                                      0.867
 lgbm 0.890 ± 0.017          0.901                                      0.902
   rf 0.889 ± 0.016          0.892                                      0.893
  mlp 0.882 ± 0.017          0.887                                      0.888
  gnb 0.860 ± 0.013          0.831                                      0.832
  knn 0.867 ± 0.010          0.869                                      0.870
```

### 2. F1, precision, recall, log loss

```
model   CV macro F1  Student macro F1  Precision  Recall  Log loss (student)
   lr 0.913 ± 0.005             0.908      0.834   0.844              0.1051
 lgbm 0.935 ± 0.011             0.940      0.906   0.884              0.0734
   rf 0.936 ± 0.010             0.937      0.915   0.866              0.0705
  mlp 0.930 ± 0.009             0.933      0.890   0.878              0.0832
  gnb 0.850 ± 0.010             0.838      0.622   0.873              0.5741
  knn 0.912 ± 0.007             0.913      0.854   0.842              0.1333
```

### 3. Overfitting check

```
model  Train Top-1  CV (validation) Top-1  Gap (train - val)            Reading
   lr        0.888                  0.881              0.008    no real overfit
 lgbm        0.932                  0.890              0.043         slight gap
   rf        0.923                  0.889              0.034         slight gap
  mlp        0.921                  0.882              0.039         slight gap
  gnb        0.871                  0.860              0.011    no real overfit
  knn          NaN                  0.867                NaN n/a (not measured)
```

### 4. Route continuity and cost

```
model  Route continuity (1-hop)  Train time (s)  Inference per trip (s)
   lr                     0.923             3.1                  0.0012
 lgbm                     0.931            11.6                  0.0232
   rf                     0.931            31.8                  0.0133
  mlp                     0.936            70.5                  0.0180
  gnb                     0.917             0.4                  0.0022
  knn                     0.924             0.1                  2.0951
```
