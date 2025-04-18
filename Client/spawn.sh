#!/bin/bash

read -p "Number of clients to run: " num
session="multi_client"

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
