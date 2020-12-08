#!/usr/bin/expect
spawn ssh edge05@192.168.0.15
expect "password:"
send "edge05\n"
expect "*$"
send "cd /data/jcliu/FL/RE-AFL/;python3 client.py --world_size 10 --dataset_type FashionMNIST --alpha 0.1 --lr 0.001 --rank 5 \n"
expect "*$"
interact
