# python main2.py --eval\
#  --mc\
#  --resume /home/vitallab/ssd/vitallab/uaCADx/mpvit_small_0.0_dropout_0.3_pretrained_best_model.pt\
#   --model mpvit_small --data-path data --output_dir new/int --drop 0.3 --metrics
  
python main2.py --eval\
 --mc --ext_val Ext_val/adhp\
 --resume /home/vitallab/ssd/vitallab/uaCADx/mpvit_small_0.0_dropout_0.3_pretrained_best_model.pt\
  --model mpvit_small --data-path data --output_dir new/adhp --drop 0.3 --metrics  --save_cam --disable_amp --batch-size 16 --mc_iter 1

python main2.py --eval\
 --mc --ext_val Ext_val/as\
 --resume /home/vitallab/ssd/vitallab/uaCADx/mpvit_small_0.0_dropout_0.3_pretrained_best_model.pt\
  --model mpvit_small --data-path data --output_dir new/as --drop 0.3 --metrics --save_cam --disable_amp --batch-size 16 --mc_iter 1

python main2.py --eval\
 --mc --ext_val Ext_val/eh\
 --resume /home/vitallab/ssd/vitallab/uaCADx/mpvit_small_0.0_dropout_0.3_pretrained_best_model.pt\
  --model mpvit_small --data-path data --output_dir new/eh --drop 0.05 --metrics --save_cam --disable_amp --batch-size 16 --mc_iter 1

# python main2.py --eval\
#  --mc --ext_val Ext_val/sp\
#  --resume /home/vitallab/ssd/vitallab/uaCADx/mpvit_small_0.0_dropout_0.3_pretrained_best_model.pt\
#   --model mpvit_small --data-path data --output_dir new/sp --drop 0.3 --metrics

# python main2.py --eval\
#  --mc --ext_val Ext_val/ssa\
#  --resume /home/vitallab/ssd/vitallab/uaCADx/mpvit_small_0.0_dropout_0.3_pretrained_best_model.pt\
#   --model mpvit_small --data-path data --output_dir new/ssa --drop 0.3 --metrics

python main2.py --eval\
 --mc --ext_val Ext_val/ys\
 --resume /home/vitallab/ssd/vitallab/uaCADx/mpvit_small_0.0_dropout_0.3_pretrained_best_model.pt\
  --model mpvit_small --data-path data --output_dir new/ys --drop 0.3 --metrics --save_cam --disable_amp --batch-size 16

