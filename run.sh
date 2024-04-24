# rm -rf /home/kevin/BAARF/out/lego/test_pred_view/*

# python baangp/train_baangp.py --scene lego --data-root ./data/nerf_synthetic/ --save-dir ./save/ --c2f 0.1 0.5
python baangp/train_baangp.py --scene lego --data-root ./data/nerf_synthetic/ --save-dir ./out/lego/ --c2f 0.1 0.5 --eps 0.0001
python baangp/train_baangp.py --scene lego --data-root ./data/nerf_synthetic/ --save-dir ./out/lego/ --c2f 0.1 0.5 --eps 0.0005
python baangp/train_baangp.py --scene lego --data-root ./data/nerf_synthetic/ --save-dir ./out/lego/ --c2f 0.1 0.5 --eps 0.001
python baangp/train_baangp.py --scene lego --data-root ./data/nerf_synthetic/ --save-dir ./out/lego/ --c2f 0.1 0.5 --eps 0.005
python baangp/train_baangp.py --scene lego --data-root ./data/nerf_synthetic/ --save-dir ./out/lego/ --c2f 0.1 0.5 --eps 0.01
python baangp/train_baangp.py --scene lego --data-root ./data/nerf_synthetic/ --save-dir ./out/lego/ --c2f 0.1 0.5 --eps 0.05
python baangp/train_baangp.py --scene lego --data-root ./data/nerf_synthetic/ --save-dir ./out/lego/ --c2f 0.1 0.5 --eps 0.1

# python baangp/train_baangp.py --scene ship --data-root ./data/nerf_synthetic/ --save-dir ./save_ship/ --c2f 0.1 0.5