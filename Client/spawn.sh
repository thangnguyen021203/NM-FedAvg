#!/bin/bash

read -p "Number of clients to run: " num
session="multi_client"

# Bật NVIDIA MPS
echo "[INFO] Starting NVIDIA MPS..."
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d
sleep 2

# Tạo tmux session mới
tmux new-session -d -s $session

# Cửa sổ đầu tiên
tmux send-keys -t $session "python3 Main.py" C-m

# Các cửa sổ tiếp theo
for ((i=2; i<=num; i++))
do
    tmux split-window -t $session -v
    tmux select-layout -t $session tiled
    tmux send-keys -t $session "python3 Main.py" C-m
done

# Gắn (attach) vào session để xem
tmux attach-session -t $session

# Sau khi thoát khỏi tmux, tắt MPS
echo "[INFO] Stopping NVIDIA MPS..."
echo quit | nvidia-cuda-mps-control
