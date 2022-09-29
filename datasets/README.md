# Data Preparing

1. Download the OCTA dataset from the [Baidu Netdisk](https://pan.baidu.com/s/1LVpg--HKS45mALSgIX5cNg?pwd=dvdj) (access code: dvdj) and decompress it to the root directory.
2. Description of the OCTA dataset:
* The dataset in folder **Data** was collected at the First Affiliated Hospital of Xi’an Jiaotong University, consisting of 110 normal eyes from 110 subjects. For each eye, both OCTA image (3×3 mm2 optic disc-centered, 320×320 pixels) and fundus image (FOV of 45°, optic disc-centered, 2576×1934 pixels) were captured. The dataset in folder **Data66** was collected at Zhongshan Ophthalmic Center of Sun Yat-sen University, consisting of 908 normal eyes from 476 subjects. For each eye, both OCTA image (6×6 mm2 optic disc-centered, 320×320 pixels) and fundus image (FOV of 45°, optic disc-centered, 1634×1634 pixels) were captured. 
* The original data and augmented data are stored in folders Origin and Augment respectively.
* data_A to data_D are the folders containing OCTA images of four different depths.
* Data_fold_1 to Data_fold_k are the sub-datasets for k-fold cross validation.
* data_Fusion containing the pseudo-color OCTA images obtained by fusing OCTA images of four depths.
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
    │    │      ├── Data_fold_1
    │    │      │      ├── train
    │    │      │      ├── train_GT
    │    │      │      ├── valid
    │    │      │      ├── valid_GT
    │    │      │      ├── test
    │    │      │      ├── test_GT
    │    │      │      ├── test_all
    │    │      │      └── test_all_GT
    │    │      ├── Data_fold_2
    │    │      │      ├── train
    │    │      │      └── ...
    │    │      ├── Data_fold_3
    │    │      ├── Data_fold_4
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
