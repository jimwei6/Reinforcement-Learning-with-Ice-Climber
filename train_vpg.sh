# Sample Usage
# python ./train_pg.py --agent VPG --dir ./pg --reward SPARSE --rgb --checkpoint ./pg/ep_1.chkpt --lr 0.001 \
# --episodes 500 --scheduler --grad-acc-size 1 --save-video-episode 50 --render
# Realistically (tune learning rate and accumulate batch size)
# python ./train_pg.py --agent VPG --dir ./pg --reward SPARSE --lr 0.001 --episodes 500 --grad-acc-size 6

# Basic step lr decay test as improvements stagnent and have high variance
python ./train_pg.py --agent VPG --dir ./pg/sparse/lr1e-3/dec50_0.5 --reward SPARSE --lr 0.001 --episodes 500 --grad-acc-size 16 --save-video-episode 800
python ./train_pg.py --agent AAC --dir ./pg/sparse/lr1e-3 --reward SPARSE --lr 0.001 --episodes 500 --grad-acc-size 16 --save-video-episode 800
