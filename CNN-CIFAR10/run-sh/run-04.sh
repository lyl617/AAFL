#!/usr/bin/expect
spawn ssh edge04@192.168.0.14
expect "password:"
send "edge04\n"
expect "*$"
send "cd /data/jcliu/FL/RE-AFL/;python3 client.py --world_size 10 --dataset_type FashionMNIST  --alpha 0.1 --lr 0.001 --rank 4 \n"
expect "*$"
interact
