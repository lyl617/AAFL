#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import subprocess
import time
import re 

TIME = 10 #监视网速的时间间隔
DEVICE = 'wlan0' #网卡名字
LOCAL = 'en' #本地语言(有的是英文
ALL = False #是否或许所有网卡的网络状态

class NetSpeed():
    def __init__(self):
       self.device = DEVICE
       self.local = LOCAL
    #p_rint 参数意为是否只是打印结果，False - 返回一个列表
    def start(self, p_rint = True):
       if not ALL:
           all_net = [self.device]
       else:
           all_net = self.get_net_name()

       if  p_rint:
           for i in all_net:
              print (i,self.get_rx_tx(device = i))
       else:
           result = []
           for j in all_net:
             res = self.get_rx_tx(device = j)
             result.append({'name':j,'rx':res[0],'tx':res[1]})
           return result

    #获取所有网卡名字
    def get_net_name(self):
       with file('/etc/udev/rules.d/70-persistent-net.rules','r')as f:
           r = f.read()
           return re.findall(r'NAME="(.+?)"',r)

    #调用系统命令ifconfig获取指定网卡名 已上传或者下载的字节大小，转换为kb
    def ifconfig(self, device = 'wlan0', local = 'en'):
       output = subprocess.Popen(['ifconfig', device], stdout=subprocess.PIPE).communicate()[0]
       output = str(output)
       if local == 'zh':
           rx_bytes = re.findall('字节 ([0-9]*) ', output)[0]
           tx_bytes = re.findall('字节 ([0-9]*) ', output)[1]
       else:
           rx_bytes = re.findall('bytes ([0-9]*)', output)[0]
           tx_bytes = re.findall('bytes ([0-9]*)', output)[1]
       return float(int(rx_bytes) / 1024), float((int(tx_bytes) / 1024))

    #获取指定网卡的网速
    def get_rx_tx(self,device = 'wlan0',local = LOCAL):
       try:
           while True:
              rx_bytes, tx_bytes = self.ifconfig(device= device,local = self.local)
            #   rx_bytes_new,tx_bytes_new = self.ifconfig(device= device,local = self.local)

            #   col_rx = (rx_bytes_new - rx_bytes)
            #   col_tx = (tx_bytes_new - tx_bytes)

            #   col_rx = '%.3f' % col_rx
            #   col_tx = '%.3f' % col_tx
              return rx_bytes, tx_bytes
       except Exception as e:
           raise e

# if __name__ == '__main__':
#     Speedp = NetSpeed()
#     rx_bytes, tx_bytes = Speedp.get_rx_tx(device = 'wlan0', local = 'en')
#     time.sleep(TIME)
#     rx_bytes_new,tx_bytes_new = Speedp.get_rx_tx(device = 'wlan0', local = 'en')
#     print ("(", rx_bytes_new - rx_bytes, ", ", (tx_bytes_new - tx_bytes), ")")
