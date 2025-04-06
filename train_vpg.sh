# Sample Usage
# python ./train_pg.py --agent VPG --dir ./pg --reward SPARSE --rgb --checkpoint ./pg/ep_1.chkpt --lr 0.001 \
# --episodes 500 --scheduler --grad-acc-size 1 --save-video-episode 50 --render
# Realistically (tune learning rate and accumulate batch size)
# python ./train_pg.py --agent VPG --dir ./pg --reward SPARSE --lr 0.001 --episodes 500 --grad-acc-size 6

# Final 
python ./train_pg.py --agent PPO --dir ./pg/sparse/lr1e-4/norm --reward SPARSE --lr 0.0001 --episodes 10000 --grad-acc-size 1 --save-video-episode 800 --rgb

