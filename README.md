# uaCADx
uncertainty-aware CADx

## Training

### MPViT

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
