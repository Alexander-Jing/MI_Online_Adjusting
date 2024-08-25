import socket
import numpy as np
import json
from progress.bar import Bar
import math
#json解析numpy数据类
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


addr = ('172.18.22.21', 8888) #设置服务端ip地址和端口号
buff_size = 65535         #消息的最大长度
tcpSerSock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
tcpSerSock.bind(addr)
tcpSerSock.listen(1)
tcpSerSock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)
while True:
    print('等待连接...')
    tcpCliSock, addr = tcpSerSock.accept()
    print('连接到:', addr)

    while True:
        decode = []                                   #存放解码后的数据
        recv_data = []
        #只接受第一个数据包的数据，提取里面的数据量值
        while not recv_data:
            recv_data = tcpCliSock.recv(int(buff_size))  # 这里是按照字节读取的，len(recv_data)返回的是字节数
        data_bytes = np.frombuffer(bytes(recv_data),count=1,dtype='>f4')
        #如果没有接收完大小为data_bytes[0]的数据，则一直接收
        while len(recv_data) < data_bytes[0]:
            data = []
            while not data:
                data = tcpCliSock.recv(int(buff_size))
            recv_data += data              #数据拼接
        data_bytes = np.frombuffer(bytes(recv_data),count=1,dtype='>f4')
        print(data_bytes,len(recv_data))
        decode = np.frombuffer(bytes(recv_data),dtype='>f4')     #数据解码                
        print('接收到数据')
        height = int(decode[1])
        width = int(decode[2])
        echo = decode[3:int(height*width+3)]
        mat = echo.reshape((height,width),order='F')
        
        #完成计算任务
        U, S, V = np.linalg.svd(mat, full_matrices=False)   #奇异值分解
        #在字典里增加一个L来保存返回数据包的大小
        result={'L':2e8,  # 以字典的形式发送数据
                'U':U,
                'S':S,
                'V':V}
        send_result = json.dumps(result,cls=NpEncoder)      #将字典编码一次，看看数据包有多大
        result['L'] = len(send_result) #把数据包真实大小写进字典
        #matlab数据包接收不完全的时候会等到超时，浪费时间。把数据包用‘ ’填满到matlab接收数据包大小的整数倍。
        matlab_buffer = 8388608
        fill_space = math.ceil(result['L']/matlab_buffer+1)*matlab_buffer
        send_result = json.dumps(result,cls=NpEncoder).ljust(fill_space).encode('utf-8')#重编码
        print('需要发回%d字节的数据包'%len(send_result))
        tcpCliSock.sendto(send_result,addr) #数据发送
        print('发送完成')
        break
    tcpCliSock.close()
tcpSerSock.close()

