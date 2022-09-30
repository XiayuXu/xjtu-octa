# AV-casNet
AV-casNet: Fully Automatic Arteriole-Venule Segmentation and Differentiation in OCT Angiography

## 1. Prepare OCTA datasets 
You can download the datasets from the [Baidu Netdisk](https://pan.baidu.com/s/1F04DSao6tLhbB5h5pUQqcQ?pwd=r5kz) (access code: r5kz) and decompress it to the root directory. Please go to ["./datasets/README.md"](datasets/README.md) for details. 

## 2. Environment

- Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3. Run the whole workflow

- the whole workflow includes training and testing a CNN model, building the graph data based on CNN predictions, and training and testing a GAT model. If you want to run the all processes automatically, you can type the following and the segmentation result will be save in the folder GAT_final:
python terminal.py --train_cnn=1 --test_cnn=1 --build_graph=1 --train_gat=1 --test_gat=1 --extract_result=1 --name='GAT_final'
- the whole workflow includes training and testing a CNN model, building the graph data based on CNN predictions, and training and testing a GAT model. The batch size we used is 4. If you do not have enough GPU memory, the bacth size can be reduced to 2.

- Train

```bash
python terminal.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
sh test.sh or python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```
## 4. Build the graph data for segmentation maps

- Train

```bash
python terminal.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)

## Citation

```bibtex
@misc{cao2021swinunet,
      title={AV-casNet: Fully Automatic Arteriole-Venule Segmentation and Differentiation in OCT Angiography}, 
      author={Xiayu Xu, Peiwei Yang, Hualin Wang, Zhanfeng Xiao, Gang Xing, Xiulan Zhang, Wei Wang, Feng Xu, Jiong Zhang, Jianqin Lei},
      year={2022},
}
```
