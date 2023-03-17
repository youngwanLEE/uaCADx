# # ViT-small
# # pretrained
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model vit_small_patch16_224 --pretrained --data-path ~/data/polyp --output ./output/vit_small_patch16_224_pretrained

# # from-scratch
# python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model vit_small_patch16_224 --data-path ~/data/polyp --output ./output/vit_small_patch16_224_from_scratch


#########################
###### mpvit-small ######
#########################
# pretrained
MODEL_NAME="mpvit_small"
EPOCH=100
OUTPUT_DIR="./output/${MODEL_NAME}_${EPOCH}ep_pretrained"
EXT_VAL="Ext_val/crop_ad_hp"

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Folder $output does not exist. Create folder and copy files ..."
  mkdir -p ${OUTPUT_DIR}
  cp $0 ${OUTPUT_DIR}/run.bash

    python -m torch.distributed.launch --nproc_per_node 8 --use_env main2.py --model ${MODEL_NAME} --epochs 100 --pretrained_mpvit "./output/pretrained/mpvit_small.pth" --data-path ~/data/polyp --ext_val ${EXT_VAL} --output ${OUTPUT_DIR} --metrics | tee -a ${OUTPUT_DIR}/history.txt

else
    echo "Folder ${OUTPUT_DIR} already exists. Please remove the folder and re-run the script."
fi

#from-scratch
OUTPUT_DIR="./output/${MODEL_NAME}_${EPOCH}ep_from_scratch"
if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Folder $output does not exist. Create folder and copy files ..."
  mkdir -p ${OUTPUT_DIR}
  cp $0 ${OUTPUT_DIR}/run.bash

    python -m torch.distributed.launch --nproc_per_node 8 --use_env main2.py --model ${MODEL_NAME} --epochs 100 --data-path ~/data/polyp --ext_val ${EXT_VAL} --output ${OUTPUT_DIR} --metrics | tee -a ${OUTPUT_DIR}/history.txt

else
    echo "Folder ${OUTPUT_DIR} already exists. Please remove the folder and re-run the script."
fi



##########################
###### inception_v3 ######
##########################

# pretrained
MODEL_NAME="inception_v3"
EPOCH=100
OUTPUT_DIR="./output/${MODEL_NAME}_${EPOCH}ep_pretrained"
EXT_VAL="Ext_val/crop_ad_hp"

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Folder $output does not exist. Create folder and copy files ..."
  mkdir -p ${OUTPUT_DIR}
  cp $0 ${OUTPUT_DIR}/run.bash

    python -m torch.distributed.launch --nproc_per_node 8 --use_env main2.py --model ${MODEL_NAME} --epochs 100 --pretrained --data-path ~/data/polyp --ext_val ${EXT_VAL} --output ${OUTPUT_DIR} --metrics | tee -a ${OUTPUT_DIR}/history.txt

else
    echo "Folder ${OUTPUT_DIR} already exists. Please remove the folder and re-run the script."
fi

#from-scratch
OUTPUT_DIR="./output/${MODEL_NAME}_${EPOCH}ep_from_scratch"
if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Folder $output does not exist. Create folder and copy files ..."
  mkdir -p ${OUTPUT_DIR}
  cp $0 ${OUTPUT_DIR}/run.bash

    python -m torch.distributed.launch --nproc_per_node 8 --use_env main2.py --model ${MODEL_NAME} --epochs 100 --data-path ~/data/polyp --ext_val ${EXT_VAL} --output ${OUTPUT_DIR} --metrics | tee -a ${OUTPUT_DIR}/history.txt

else
    echo "Folder ${OUTPUT_DIR} already exists. Please remove the folder and re-run the script."
fi


##########################
###### inception_v3 ######
##########################

# pretrained
MODEL_NAME="inception_v3"
EPOCH=100
OUTPUT_DIR="./output/${MODEL_NAME}_${EPOCH}ep_pretrained"
EXT_VAL="Ext_val/crop_ad_hp"

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Folder $output does not exist. Create folder and copy files ..."
  mkdir -p ${OUTPUT_DIR}
  cp $0 ${OUTPUT_DIR}/run.bash

    python -m torch.distributed.launch --nproc_per_node 8 --use_env main2.py --model ${MODEL_NAME} --epochs 100 --pretrained --data-path ~/data/polyp --ext_val ${EXT_VAL} --output ${OUTPUT_DIR} --metrics | tee -a ${OUTPUT_DIR}/history.txt

else
    echo "Folder ${OUTPUT_DIR} already exists. Please remove the folder and re-run the script."
fi

#from-scratch
OUTPUT_DIR="./output/${MODEL_NAME}_${EPOCH}ep_from_scratch"
if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "Folder $output does not exist. Create folder and copy files ..."
  mkdir -p ${OUTPUT_DIR}
  cp $0 ${OUTPUT_DIR}/run.bash

    python -m torch.distributed.launch --nproc_per_node 8 --use_env main2.py --model ${MODEL_NAME} --epochs 100 --data-path ~/data/polyp --ext_val ${EXT_VAL} --output ${OUTPUT_DIR} --metrics | tee -a ${OUTPUT_DIR}/history.txt

else
    echo "Folder ${OUTPUT_DIR} already exists. Please remove the folder and re-run the script."
fi