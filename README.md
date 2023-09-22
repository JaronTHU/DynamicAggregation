

# Fine-Tuning Point Cloud Transformers with Dynamic Aggregation

The official implementation of DA, based on the official implementation of Point-MAE. Slight modification.

## 1. Installation

```
conda create -n pt python=3.7
conda activate pt
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
cd ./extensions/chamfer_dist
python setup.py install --user
cd ../../
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```


## 2. Datasets

We use ShapeNet, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. See [DATASET.md](./DATASET.md) for details.

## 3. fine-tune ckpt

All pretrain models are in `reserve` directory.

```
.
├── fewshot_way_10_shot_10
│   ├── Adapter
│   ├── Bias
│   ├── DA-Heavy
│   ├── DA-Light
│   ├── Full
│   ├── Head
│   ├── result.csv
│   ├── VPT-Deep
│   └── VPT-Shallow
├── fewshot_way_10_shot_20
│   ├── Adapter
│   ├── Bias
│   ├── DA-Heavy
│   ├── DA-Light
│   ├── Full
│   ├── Head
│   ├── result.csv
│   ├── VPT-Deep
│   └── VPT-Shallow
├── fewshot_way_5_shot_10
│   ├── Adapter
│   ├── Bias
│   ├── DA-Heavy
│   ├── DA-Light
│   ├── Full
│   ├── Head
│   ├── result.csv
│   ├── VPT-Deep
│   └── VPT-Shallow
├── fewshot_way_5_shot_20
│   ├── Adapter
│   ├── Bias
│   ├── DA-Heavy
│   ├── DA-Light
│   ├── Full
│   ├── Head
│   ├── result.csv
│   ├── VPT-Deep
│   └── VPT-Shallow
├── modelnet
│   ├── Adapter
│   ├── Bias
│   ├── DA-Heavy
│   ├── DA-Light
│   ├── Full
│   ├── Head
│   ├── result.csv
│   ├── Scratch
│   ├── VPT-Deep
│   └── VPT-Shallow
├── pretrain.pth
├── scan_hardest
│   ├── Adapter
│   ├── Bias
│   ├── DA-Heavy
│   ├── DA-Light
│   ├── Full
│   ├── Head
│   ├── result.csv
│   ├── Scratch
│   ├── VPT-Deep
│   └── VPT-Shallow
├── scan_objbg
│   ├── Adapter
│   ├── Bias
│   ├── DA-Heavy
│   ├── DA-Light
│   ├── Full
│   ├── Head
│   ├── result.csv
│   ├── Scratch
│   ├── VPT-Deep
│   └── VPT-Shallow
├── scan_objonly
│   ├── Adapter
│   ├── Bias
│   ├── DA-Heavy
│   ├── DA-Light
│   ├── Full
│   ├── Head
│   ├── result.csv
│   ├── Scratch
│   ├── VPT-Deep
│   └── VPT-Shallow
└── shapenetpart
    ├── Adapter
    ├── Bias
    ├── DA-Light
    ├── DA-Naive
    ├── Full
    ├── Head
    ├── result.csv
    ├── Scratch
    ├── VPT-Deep
    └── VPT-Shallow
```

## 4. DA Fine-tuning

ModelNet (Scan_*), run:
```
# fine-tune
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config reserve/modelnet/DA-Light/config.yaml --finetune_model --exp_name <output_file_name> --ckpts reserve/pretrain.pth

# test
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config reserve/modelnet/DA-Light/config.yaml --test --exp_name <output_file_name> --ckpts reserve/modelnet/DA-Light/ckpt-best.pth
```

Few-shot learning, run:
```
# train
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config reserve/fewshot_way_5_shot_10/DA-Light/fold_0/config.yaml --finetune_model --ckpts reserve/pretrain.pth --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```

Part segmentation on ShapeNetPart, run:
```
# fine-tune
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config reserve/shapenetpart/DA-Light/config.yaml --partseg --exp_name <output_file_name> --ckpts reserve/pretrain.pth

# test
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config reserve/shapenetpart/DA-Light/config.yaml --test_partseg --exp_name <output_file_name> --ckpts reserve/shapenetpart/DA-Light/ckpt-best.pth
```

## 5. Example commands on multiple GPUs

Single GPU / Mutiple GPUs (DataParallel)
```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config <cfg_path> --exp_name <output_file_name>
```

Multiple GPUs (DistributedDataParallel)
```
CUDA_VISIBLE_DEVICES=<GPU> torchrun --nproc_per_node <num_of_gpus> main.py --config <cfg_path> --exp_name <output_file_name> --launcher pytorch
```

Test
```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/finetune/modelnet.yaml --test --exp_name <output_file_name>
```


## Acknowledgements

Our codes are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE).

## Reference

