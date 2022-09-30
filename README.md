# AV-casNet
AV-casNet: Fully Automatic Arteriole-Venule Segmentation and Differentiation in OCT Angiography

## 1. Prepare OCTA datasets 
You can download the datasets from the [Baidu Netdisk](https://pan.baidu.com/s/1F04DSao6tLhbB5h5pUQqcQ?pwd=r5kz) (access code: r5kz) and decompress it to the root directory. Please go to ["./datasets/README.md"](datasets/README.md) for details. 

## 2. Environment

- Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3. Train/Test

- Run the train script on synapse dataset. The batch size we used is 24. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train

```bash
sh train.sh or python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
sh test.sh or python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)

## Citation

```bibtex
@misc{cao2021swinunet,
      title={Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation}, 
      author={Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
      year={2021},
      eprint={2105.05537},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
