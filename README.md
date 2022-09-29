# xjtu-octa
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
    │    │             ├── train
    │    │             ├── train_GT
    │    │             ├── valid
    │    │             ├── valid_GT
    │    │             ├── test
    │    │             ├── test_GT
    │    │             ├── test_all
    │    │             └── test_all_GT
    │    │      ├── Data66_fold_2
    │    │             ├── train
    │    │             └── ...
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
