# Sample Command
# python ./train_dqn.py --agent DQN --dir ./DQN --checkpoint ./DQN/stored_checkpoint.chkpt --lr 0.0001 \
# --reward SPARSE --rgb --episodes 1000 --memory 512 --batch 64 --start-learning 128 --decay 1000 --render

# Realistically (Tune the lr, memory, batch, and decay):
# python ./train_dqn.py --agent DQN --dir ./DQN --lr 0.0001 --reward SPARSE --episodes 1000 --memory 512 --batch 64 \
# --start-learning 128 --decay 1000 --save-video-episode 100 

# python ./train_dqn.py --agent DQN --dir ./DQN/int/lr1e-3 --lr 0.001 --reward INT --episodes 2000 --memory 25000 --batch 256 --decay 40000 --save-video-episode 100 --start-learning 1024
# python ./train_dqn.py --agent NstepDuelingDQN --dir ./DQN/rewards4/lr1e-3 --lr 0.001 --reward SPARSE --episodes 2000 --memory 25000 --batch 256 --decay 40000 --save-video-episode 100 --start-learning 1024
# python ./train_dqn.py --agent DuelingDQN --dir ./DQN/rewards2/lr1e-4 --lr 0.0001 --reward SPARSE --episodes 2000 --memory 16384 --batch 128 --decay 40000 --save-video-episode 100 --start-learning 1024
# python ./train_dqn.py --agent PRDQN --dir ./DQN/lr1e-4 --lr 0.0001 --reward SPARSE --episodes 1000 --memory 16384 --batch 128 --decay 20000 --save-video-episode 100 --start-learning 1024
# python ./train_dqn.py --agent DuelingPRDQN --dir ./DQN/lr1e-4 --lr 0.0001 --reward SPARSE --episodes 1000 --memory 16384 --batch 128 --decay 20000 --save-video-episode 100 --start-learning 1024
# python ./train_dqn.py --agent NstepDuelingPRDQN --dir ./DQN/lr1e-4 --lr 0.0001 --reward SPARSE --episodes 1000 --memory 16384 --batch 128 --decay 20000 --save-video-episode 100 --start-learning 1024

# Test NstepDuelingPRDQN on Best

python ./train_dqn.py --agent NstepDuelingPRDQN --dir ./DQN/int/norm/lr1e-3 --lr 0.001 --reward INT --episodes 2000 --memory 25000 --batch 128 --decay 40000 --save-video-episode 100 --start-learning 1024 --rgb