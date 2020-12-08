#!/usr/bin/expect
spawn ssh edge01@192.168.0.11
expect "password:"
send "edge01\n"
expect "*$"
send "cd /data/jcliu/FL/RE-AFL/;python3.6 client.py --world_size 10 --dataset_type FashionMNIST --alpha 0.1 --lr 0.001 --rank 1 \n"
expect "*$"
interact
