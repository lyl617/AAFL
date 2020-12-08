#!/usr/bin/expect
spawn ssh edge07@192.168.0.17
expect "password:"
send "edge07\n"
expect "*$"
send "cd /data/jcliu/FL/RE-AFL/;python3 client.py --world_size 10 --dataset_type FashionMNIST --alpha 0.1 --lr 0.001 --rank 7 \n"
expect "*$"
interact
