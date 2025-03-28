# Sample Command
# python ./train_dqn.py --agent DQN --dir ./DQN --checkpoint ./DQN/stored_checkpoint.chkpt --lr 0.0001 \
# --reward SPARSE --rgb --episodes 1000 --memory 512 --batch 64 --decay 1000 --buffer-device cuda --render

# Realistically (Tune the lr, memory, batch, and decay):
# python ./train_dqn.py --agent DQN --dir ./DQN --lr 0.0001 --reward SPARSE --episodes 1000 --memory 512 --batch 64 --decay 1000 --buffer-device cuda

python ./train_dqn.py --agent DQN --dir ./DQN --lr 0.0001 --reward SPARSE --episodes 1000 --memory 512 --batch 64 --decay 1000
python ./train_dqn.py --agent PRDQN --dir ./DQN --lr 0.0001 --reward SPARSE --episodes 1000 --memory 512 --batch 64 --decay 1000
python ./train_dqn.py --agent DuelingPRDQN --dir ./DQN --lr 0.0001 --reward SPARSE --episodes 1000 --memory 512 --batch 64 --decay 1000
python ./train_dqn.py --agent NstepDuelingPRDQN --dir ./DQN --lr 0.0001 --reward SPARSE --episodes 1000 --memory 512 --batch 64 --decay 1000