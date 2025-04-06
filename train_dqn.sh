# Sample Command
# python ./train_dqn.py --agent DQN --dir ./DQN --checkpoint ./DQN/stored_checkpoint.chkpt --lr 0.0001 \
# --reward SPARSE --rgb --episodes 1000 --memory 512 --batch 64 --start-learning 128 --decay 1000 --render

# Final
python ./train_dqn.py --agent NstepDuelingPRDQN --dir ./DQN/int/norm/lr1e-3 --lr 0.001 --reward INT --episodes 2000 --memory 25000 --batch 128 --decay 40000 --save-video-episode 100 --start-learning 1024 --rgb