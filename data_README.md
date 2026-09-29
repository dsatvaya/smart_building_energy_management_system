# Data

The CU-BEMS dataset is **not** tracked by git (~733 MB). Download it here before running the pipeline.

## Option A — Kaggle CLI

```bash
pip install kaggle
# place your kaggle.json in ~/.kaggle/ first
kaggle datasets download -d claytonmiller/cubems-smart-building-energy-and-iaq-data
unzip cubems-smart-building-energy-and-iaq-data.zip -d .
```

## Option B — manual

Download from the Kaggle dataset page and unzip into this folder.

## Expected contents

```
data/
├── 2018Floor1.csv  ...  2018Floor7.csv
└── 2019Floor1.csv  ...  2019Floor7.csv
```

If disk space is tight, `2019Floor6.csv` alone is enough to reproduce the baseline results.

## Citation

Pipattanasomporn, M., Chitalia, G., Songsiri, J., Aswakul, C., Pora, W., Suwankawin, S., Audomvongseree, K., Hoonchareon, N. (2020). CU-BEMS, smart building electricity consumption and indoor environmental sensor datasets. *Scientific Data* 7:1.
