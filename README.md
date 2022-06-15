# uaCADx
uncertainty-aware CADx

## Training

### MPViT

download the pretrained weight of mpvit-small
```
wget https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth
```

```
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model mpvit_small --pretrained_mpvit <path/to/pretrained_ckpt> --data-path <path/to/polyp> --output <path/to/output_dir>
```

### DeiT

```
 python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model deit_small_patch16_224 --pretrained --data-path <path/to/polyp> --output <path/to/output_dir>
```

### ViT


```
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model vit_small_patch16_224 --pretrained --data-path <path/to/polyp> --output <path/to/output_dir>
```


## Baseline

|     Model                                	|     MPViT-Small    	|     ViT-Small        	|     DeiT-Small       	|     ResNet-50        	|
|------------------------------------------	|:------------------:	|:---------------------:|:---------------------:|:---------------------:|
|     Model size                           	|     22.8M          	|     22.1M            	|     22.1M            	|     25.6M            	|
|     Accuracy (Last top-1/ Max top-1)     	|     97.0 / 96.0    	|     93.7 /   92.0    	|     93.0 /   92.3    	|     95.0 /   94.3    	|
