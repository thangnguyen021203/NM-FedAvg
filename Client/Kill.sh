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
else
    echo "Đã hủy lệnh."
fi
