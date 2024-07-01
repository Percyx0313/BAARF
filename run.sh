export CUDA_VISIBLE_DEVICES=0
# SCENE_LIST=( "materials" "ship"     "drums"  "chair"  "mic"  "ficus"  "lego"  "hotdog" )  # 
# SCENE_LIST=( "teapot" "helmet" "car" "coffee" "toaster" "ball")  # 
SCENE_LIST=( "horns" ) #"fern" "trex"   "flower"  "orchids"   "leaves"         "fortress"  "room")  


GROUP="LLFF_matching"
mkdir -p out/${GROUP}
for SCENE in "${SCENE_LIST[@]}"
do
echo "training ${SCENE}"
mkdir out/${GROUP}/${SCENE}
# python baangp/train_baangp.py --scene ${SCENE} --data-root baangp/datasets/llff --save-dir out/LLFF/${SCENE} --c2f 0.1 0.5 --dataset llff # --wandb
python -u baangp/train_baangp.py --scene ${SCENE} --data-root baangp/datasets/llff --save-dir out/${GROUP}/${SCENE} --c2f 0.1 0.5 --dataset llff \
 --gt_camera | tee -a  out/${GROUP}/${SCENE}/log.txt #--wandb
done
# --optim_pose

# GROUP="test"
# mkdir -p out/${GROUP}
# SCENE_LIST=("g")  
# for SCENE in "${SCENE_LIST[@]}"
# do
# echo "training ${SCENE}"
# mkdir out/${GROUP}/${SCENE}
# # python baangp/train_baangp.py --scene ${SCENE} --data-root baangp/datasets/llff --save-dir out/LLFF/${SCENE} --c2f 0.1 0.5 --dataset llff # --wandb
# python -u baangp/train_baangp.py --scene ${SCENE} --data-root baangp/datasets/mipnerf360 --save-dir out/${GROUP}/${SCENE} --c2f 0.1 0.5 --dataset llff --gt_camera  \
#  --optim_pose | tee -a  out/${GROUP}/${SCENE}/log.txt #--wandb 
# done
