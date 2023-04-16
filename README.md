# Multimodal Tree Decoder for Table of Contents Extraction in Document Images
This repository contains the source code of: [Multimodal Tree Decoder for Table of Contents Extraction in Document Images](https://ieeexplore.ieee.org/abstract/document/9956301/).

## Requirements
To execute this code, it is mandatory to prepare the following:
* Bert Model
* Pretrained ResNet-34 weights
* The proposed dataset [HierDoc](https://drive.google.com/file/d/10oFqigjt73GWc7UxPxJXDSjz1IkGTPe1/view?usp=sharing)

The Bert Model is available [here](https://github.com/huggingface/transformers). We recommend pretraining the ResNet-34 on scientific papers with a text detection task.

## Training
```shell
python runner/train_valid.py --cfg default --visual_pretrain_weights path_to_pretrained_renet34_weights
```

## Testing
```shell
python runner/infer.py --cfg default
```



## Citation
If you find our paper useful in your research, please consider citing:

```
@INPROCEEDINGS{9956301,
  author={Hu, Pengfei and Zhang, Zhenrong and Zhang, Jianshu and Du, Jun and Wu, Jiajia},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)}, 
  title={Multimodal Tree Decoder for Table of Contents Extraction in Document Images}, 
  year={2022},
  volume={},
  number={},
  pages={1756-1762},
  doi={10.1109/ICPR56361.2022.9956301}}

```

