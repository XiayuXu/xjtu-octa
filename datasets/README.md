# Data Preparing

1. Access to the synapse multi-organ dataset:
   1. Sign up in the [official Synapse website](https://www.synapse.org/#!Synapse:syn3193805/wiki/) and download the dataset. Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
   2.  You can also send an Email directly to jienengchen01 AT gmail.com to request the preprocessed data for reproduction.
2. The directory structure of the whole project is as follows:

```bash
Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@XiayuXu 
XiayuXu
/
xjtu-octa
Public
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
xjtu-octa/datasets/README
@XiayuXu
XiayuXu Create README
Latest commit 49b0a86 2 hours ago
 History
 1 contributor
42 lines (42 sloc)  1.46 KB

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
Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
xjtu-octa/README at main · XiayuXu/xjtu-octa
```
