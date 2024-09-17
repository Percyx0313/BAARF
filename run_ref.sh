export CUDA_VISIBLE_DEVICES=0
SCENE_LIST=(   "teapot" "helmet" "car" "coffee" "toaster" "ball")  # 

for SCENE in "${SCENE_LIST[@]}"
do
echo "training ${SCENE}"
python baangp/train_baangp.py --scene ${SCENE} --data-root baangp/datasets/refnerf --save-dir out/relocalization_noend/${SCENE} --c2f 0.1 0.5  --wandb
done
