#!/bin/bash
# 持续监控脚本

while true; do
    ./scripts/check_and_advance.sh
    sleep 300  # 每5分钟检查一次
done
