#############
### mpvit ###
#############
MODEL="mpvit_small"


# # snuh: original
# DATA="Int_val"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth  --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics



# # # snuh: crop_ad_hp
# DATA="Ext_val/crop_ad_hp"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics

# # ys
# DATA="Ext_val/ys"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics


# # ew
# DATA="Ext_val/ew"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics


# # as
# DATA="Ext_val/as"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics



# #############
# ### ViT #####
# #############

# MODEL="vit_small_patch16_224"

# # # snuh: original
# DATA="Int_val"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth  --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics


# # snuh: crop_ad_hp
# DATA="Ext_val/crop_ad_hp"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics

# # ys
# DATA="Ext_val/ys"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics


# # ew
# DATA="Ext_val/ew"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics


# # as
# DATA="Ext_val/as"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics



# #############
# ### Inception_v3 #####
# #############

MODEL="inception_v3"

# snuh: original
DATA="Int_val"

#pretrained
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth  --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

#from-scratch
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics

# # snuh: crop_ad_hp
# DATA="Ext_val/crop_ad_hp"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics

# # ys
# DATA="Ext_val/ys"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics


# # ew
# DATA="Ext_val/ew"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics


# # as
# DATA="Ext_val/as"

# #pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_pretrained/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_pretrained/${DATA} --data-path ~/data/polyp/ --metrics

# #from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model ${MODEL} --eval --resume ./output/${MODEL}_from_scratch/best_model.pth --ext_val ${DATA} --output_dir ./output/${MODEL}_from_scratch/${DATA} --data-path ~/data/polyp/ --metrics