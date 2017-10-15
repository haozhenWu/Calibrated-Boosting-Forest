Example script for benchmarking PCBA128 using self-contained molecules.
Instead of featuring molecules of each target, featured concatenated molecules
one time and use each target's own molecules to build models.

Usage:

```bash
$ chmod 755 ./run.sh
$ ./run.sh pcba128_TargetName.csv
```

Results: ROCAUC

Stratified split, 75% cross validation, 25% test

|         |mean|median|
|---------|----|------|
|cv result|0.891|0.914|
|test result|0.895|0.918|

Ensemble improvement  
diff = best second layer model - best first layer model

|         |mean|median|
|---------|----|------|
|cv diff|0.042|0.0.037|
|test diff|0.023|0.022|
