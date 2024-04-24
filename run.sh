# export CUDA_VISIBLE_DEVICES=1
SCENE_LIST=( "hotdog" ) # "ship" "materials"    "drums"  "chair"  "mic"    "ficus"  "lego"  ) 
for SCENE in "${SCENE_LIST[@]}"
do
echo "training ${SCENE}"
python baangp/train_baangp.py --scene ${SCENE} --data-root baangp/datasets/blender --save-dir out/relocalization/${SCENE} --c2f 0.1 0.5  #--wandb
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