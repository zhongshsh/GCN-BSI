nohup python -u /home/user02/zss/ngcf/NGCF/test.py --data_path /home/user02/zss/ngcf/Data/data/ --dataset steamgame --regs [1e-5] --embed_size 128 --layer_size [128,128,128] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 100 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] > /home/user02/zss/ngcf/NGCF/log/0519test.txt 2>&1 &


nohup python -u /home/user02/zss/ngcf/NGCF/test.py --data_path /home/user02/zss/ngcf/Data/data/ --dataset newNetease --regs [1e-5] --embed_size 128 --layer_size [128,128,128] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 100 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] > /home/user02/zss/ngcf/NGCF/log/0519test.txt 2>&1 &

nohup python -u /home/user02/zss/ngcf/NGCF/testGCMC.py --data_path /home/user02/zss/ngcf/Data/data/ --dataset steamgame --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 100 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] > /home/user02/zss/ngcf/NGCF/log/0519test.txt 2>&1 &


nohup python -u /home/user02/zss/ngcf/NGCF/testGCMC.py --data_path /home/user02/zss/ngcf/Data/data/ --dataset newNetease --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 100 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] > /home/user02/zss/ngcf/NGCF/log/0519test.txt 2>&1 &


nohup python -u /data1/jianbin/bundle_rec/neural_graph_collaborative_filtering/NGCF/testNMF.py --data_path /data1/jianbin/bundle_rec/data/ --dataset steamGame --regs [1e-5] --embed_size 256 --layer_size [256,128,32] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 100 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] > /data1/jianbin/bundle_rec/neural_graph_collaborative_filtering/NGCF/0519test.txt 2>&1 &