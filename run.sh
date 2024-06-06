export CUDA_VISIBLE_DEVICES=0
# SCENE_LIST=( "materials" "ship"     "drums"  "chair"  "mic"  "ficus"  "lego"  "hotdog" )  # 
# SCENE_LIST=( "teapot" "helmet" "car" "coffee" "toaster" "ball")  # 
SCENE_LIST=( "horns" "orchids" "fern"  "trex" "leaves"      "flower"  "room" "fortress")  


GROUP="LLFF_warmup2"
mkdir -p out/${GROUP}
for SCENE in "${SCENE_LIST[@]}"
do
echo "training ${SCENE}"
mkdir out/${GROUP}/${SCENE}
# python baangp/train_baangp.py --scene ${SCENE} --data-root baangp/datasets/llff --save-dir out/LLFF/${SCENE} --c2f 0.1 0.5 --dataset llff # --wandb
python -u baangp/train_baangp.py --scene ${SCENE} --data-root baangp/datasets/llff --save-dir out/${GROUP}/${SCENE} --c2f 0.0 0.5 --dataset llff \
 --optim_pose | tee -a  out/${GROUP}/${SCENE}/log.txt #--wandb
done

# python baangp/train_baangp.py --scene "lego" --data-root baangp/datasets/blender --save-dir out/test --c2f 0.1 0.5

# SCENE_LIST=(  "drums"   "ship"   "mic"  "ficus" "chair" "hotdog" "lego") #   "materials" "car" "coffee"  "helmet" "teapot" "toaster" "ball" )
# #
# # GROUP="123"
# GROUP="hybarf"
# mkdir ./out/${GROUP}
# export CUDA_VISIBLE_DEVICES=1

# for SCENE in "${SCENE_LIST[@]}"
# do
#     python baangp/train_baangp.py --scene ${SCENE} --data-root baangp/datasets/blender  --save-dir ./out/${GROUP}/${SCENE} --c2f 0.1 0.5 --outlier-filter  >> ./out/${GROUP}/${SCENE}.log

#     # python baangp/train_baangp.py --scene ${SCENE} --data-root baangp/datasets/blender --outlier_thresh 95 --save-dir ./out/${GROUP}/${SCENE} --c2f 0.1 0.5  #>> ./out/${GROUP}/${SCENE}.log
# done
