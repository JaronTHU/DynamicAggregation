## Dataset

The overall directory of datasets should include:
```
├── datasets
│   ├── ModelNet40PointBERT
│   ├── ModelNetFewshot
│   ├── ScanObjectNN
│   ├── ShapeNet55-34
│   └── ShapeNetPartNormal
```

* For convenience, we place datasets out of the Point-MAE-Prompt directory

### ModelNet40 Dataset: 

```
├── ModelNet40PointBERT
│   ├── modelnet40_shape_names.txt
│   ├── modelnet40_test_8192pts_fps.dat
│   ├── modelnet40_test.txt
│   ├── modelnet40_train_8192pts_fps.dat
│   └── modelnet40_train.txt
```
Download: You can download the processed data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), or download from the [official website](https://modelnet.cs.princeton.edu/#) and process it by yourself.

### ModelNet Few-shot Dataset:
```
│ModelNetFewshot/
├──5way10shot/
│  ├── 0.pkl
│  ├── ...
│  ├── 9.pkl
├──5way20shot/
│  ├── ...
├──10way10shot/
│  ├── ...
├──10way20shot/
│  ├── ...
```

Download: Please download the data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). We use the same data split as theirs.

### ScanObjectNN Dataset:
```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```
Download: Please download the data from the [official website](https://hkust-vgd.github.io/scanobjectnn/).

### ShapeNet55/34 Dataset:

```
│ShapeNet55-34/
├──shapenet_pc/
│  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│  ├── .......
├──ShapeNet-55/
│  ├── train.txt
│  └── test.txt
```

Download: Please download the data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md).

### ShapeNetPart Dataset:

```
└── ShapeNetPartNormal
    ├── 02691156
    ├── 02773838
    ├── 02954340
    ├── 02958343
    ├── 03001627
    ├── 03261776
    ├── 03467517
    ├── 03624134
    ├── 03636649
    ├── 03642806
    ├── 03790512
    ├── 03797390
    ├── 03948459
    ├── 04099429
    ├── 04225987
    ├── 04379243
    ├── processed
    ├── synsetoffset2category.txt
    └── train_test_split
```

Download: Please download the data from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). 
