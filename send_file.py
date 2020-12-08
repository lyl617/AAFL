import paramiko
import argparse
from scp import SCPClient

parser = argparse.ArgumentParser(description='send file to the specified edge nodes')

parser.add_argument('--local_path', type=str,  metavar='N',
                        help='local file path')
parser.add_argument('--remote_path', type=str,  metavar='N',
                        help='remote path to upload')
parser.add_argument('--dst', type=str, default='all', metavar='N',
                        help='on which device to upload (default on all edge nodes)')
args = parser.parse_args()

local_path = args.local_path
remote_path = args.remote_path
dst = args.dst

def upload_img(host, user_name, passwd):
 
    s=paramiko.SSHClient()
    s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    s.connect(host,username=user_name,password=passwd)

    scpclient = SCPClient(s.get_transport(),socket_timeout=15.0)
    try:
        scpclient.put(local_path, remote_path, True)
        scpclient
    except FileNotFoundError as e:
        print(e)
        print("file not found " + local_path)
    else:
        print("file was uploaded to", user_name, ": ", host)
    scpclient.close()
    s.close()


if dst == 'all':
    node_index = [i+1 for i in range(10)]
else:
    node_index = [int(i) for i in dst.split(',')]

for index in node_index:
    host='192.168.0.'+str(index + 10)
    user_name='edge' + "{:0>2d}".format(index)
    passwd='edge' + "{:0>2d}".format(index)

    upload_img(host, user_name, passwd)
