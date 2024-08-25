import socket
 
# 明确配置变量
ip_port = ('172.18.22.21', 8888)
back_log = 5
buffer_size = 1024
# 创建一个TCP套接字
ser = socket.socket(socket.AF_INET,socket.SOCK_STREAM)   # 套接字类型AF_INET, socket.SOCK_STREAM   tcp协议，基于流式的协议
ser.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)  # 对socket的配置重用ip和端口号
# 绑定端口号
ser.bind(ip_port)  #  写哪个ip就要运行在哪台机器上
# 设置半连接池
ser.listen(back_log)  # 最多可以连接多少个客户端
while 1:
    # 阻塞等待，创建连接
    con,address = ser.accept()  # 在这个位置进行等待，监听端口号
    if con:
        print("successfully connected, the client is: ", address)
        while con:
            try:
                # 接受套接字的大小，怎么发就怎么收
                msg = con.recv(buffer_size)
                if msg.decode('utf-8') == '1':
                    # 断开连接
                    con.close()
                print(msg.decode('utf-8'))
            except Exception as e:
                break
# 关闭服务器
ser.close()