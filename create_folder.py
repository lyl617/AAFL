import os
import argparse
import paramiko
from pathlib import Path


parser = argparse.ArgumentParser(description='create folder on the specified edge nodes')

parser.add_argument('--fname', type=str,  metavar='N',
                        help='folder name to create at /home/edgexx/data')
parser.add_argument('--dst', type=str, default='all', metavar='N',
                        help='on which device to create (default on all edge nodes)')

args = parser.parse_args()
fname = args.fname
dst = args.dst

if dst == 'all':
	node_index = [i+1 for i in range(10)]
else:
	node_index = [int(i) for i in dst.split(',')]

for index in node_index:
	ab_path = "/data/" + fname

	host='192.168.0.'+str(index + 10)
	user_name='edge' + "{:0>2d}".format(index)
	passwd='edge' + "{:0>2d}".format(index)
	s=paramiko.SSHClient()
	s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	s.connect(host,username=user_name,password=passwd)

	#stdin,stdout,stderr=s.exec_command('/root/test.sh')
	if not Path("/data/").exists():
		s.exec_command('mkdir /data')
	if not Path('/data'+fname).exists():
		s.exec_command('mkdir /data/'+fname)
		print("create", fname, "on ", user_name, ":", host)
	s.close()
