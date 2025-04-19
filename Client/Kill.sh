#!/bin/bash

# Hiển thị các session đang chạy (tùy chọn)
echo "Các tmux session hiện tại:"
tmux ls

# Hỏi người dùng xác nhận
read -p "Bạn có chắc chắn muốn kill toàn bộ tmux sessions không? (y/n): " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    # Kill tất cả session tmux
    tmux kill-server
    echo "Đã dừng toàn bộ tmux sessions."

    # Tắt NVIDIA MPS nếu đang chạy
    echo "[INFO] Đang kiểm tra và dừng NVIDIA MPS nếu cần..."
    control_pipe="/tmp/nvidia-mps/control"
    if [[ -e "$control_pipe" ]]; then
        echo quit | nvidia-cuda-mps-control
        echo "Đã dừng NVIDIA MPS daemon."
    else
        echo "MPS không đang chạy hoặc control pipe không tồn tại."
    fi
else
    echo "Đã hủy lệnh."
fi
