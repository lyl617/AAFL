rank=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10")
ip=("11" "12" "13" "14" "15" "16" "17" "18" "19" "20")
port='21110'
epoch='10000'
alpha='0.0'
datanum_idx='0'
pattern_idx='2'
gnome-terminal -x bash -c "./run_client.sh edge01@192.168.0.11 edge01 'cd /data/jcliu/FL/AAFL/;python3.6 client.py --world_size 10 --dataset_type FashionMNIST  --model_type CNN --alpha $alpha --lr 0.0001 --datanum_idx $datanum_idx --pattern_idx $pattern_idx --epochs $epoch --rank 1 --port $port';exec bash;"
for i in 1 2 3 4 5 6 7 8 9
do
gnome-terminal -x bash -c "./run_client.sh edge${rank[$i]}@192.168.0.${ip[$i]} edge${rank[$i]} 'cd /data/jcliu/FL/AAFL/;python3 client.py --world_size 10 --dataset_type FashionMNIST --model_type CNN --lr 0.0001 --datanum_idx $datanum_idx --pattern_idx $pattern_idx --alpha $alpha --epochs $epoch --rank ${rank[$i]} --port $port';exec bash;"
done
gnome-terminal -x bash -c "source /etc/profile;~/anaconda3/envs/torch1.6/bin/python3 /data/jcliu/AAFL/server_thread.py --world_size 10 --dataset_type FashionMNIST --model_type CNN --alpha $alpha --lr 0.0001 --datanum_idx $datanum_idx --pattern_idx $pattern_idx --epochs $epoch --rank 0 --port $port;exec bash;"
