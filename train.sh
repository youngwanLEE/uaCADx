# ViT-small
# pretrained

python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model vit_small_patch16_224 --pretrained --data-path ~/data/polyp --output ./output/vit_small_patch16_224_pretrained

# from-scratch
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model vit_small_patch16_224 --data-path ~/data/polyp --output ./output/vit_small_patch16_224_from_scratch