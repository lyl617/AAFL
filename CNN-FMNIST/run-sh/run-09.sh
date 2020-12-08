#!/usr/bin/expect
spawn ssh edge09@192.168.0.19
expect "password:"
send "edge09\n"
expect "*$"
send "cd /data/jcliu/FL/RE-AFL/;python3 client.py --world_size 10 --dataset_type FashionMNIST --alpha 0.1 --lr 0.001 --rank 9 \n"
expect "*$"
interact
