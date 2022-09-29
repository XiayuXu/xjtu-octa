# Data Preparing

1. Download the OCTA dataset from the Baidu web disk and decompress it to the root directory.
2. Description of the OCTA dataset:
* OCTA Data of 3×3 mm2 and 6×6 mm2 are stored in Data and Data66 respectively.

3. The directory structure of the whole project is as follows:

```bash
.
└── datasets
    ├──Data66
    │    ├── Augment
    │    │      ├── data_A
    │    │      ├── data_B
    │    │      ├── data_C
    │    │      ├── data_D
    │    │      └── label
    │    ├── Origin
    │    │      ├── data_A
    │    │      ├── data_B
    │    │      ├── data_C
    │    │      ├── data_D
    │    │      ├── Data66_fold_1
    │    │      │      ├── train
    │    │      │      ├── train_GT
    │    │      │      ├── valid
    │    │      │      ├── valid_GT
    │    │      │      ├── test
    │    │      │      ├── test_GT
    │    │      │      ├── test_all
    │    │      │      └── test_all_GT
    │    │      ├── Data66_fold_2
    │    │      │      ├── train
    │    │      │      └── ...
    │    │      ├── Data66_fold_3
    │    │      ├── Data66_fold_4
    │    │      ├── Label_GRAY
    │    │      ├── Label_RGB
    │    │      └── data_Fusion
    │    │
    │    ├── annotation2label.py
    │    ├── augment.py
    │    ├── DataMaker_Cross_valid.py
    │    └── fusion.py
    └──Data
         ├── Augment
         │     └── ...
         ├── Origin
         │     └── ...
         └── ...

```
