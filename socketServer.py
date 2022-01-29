import socket
import time 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 9090 ))


try:
    s.connect(('0.0.0.0', 9090 ))
except:
    print("Connection failed")


s.listen(0)                 
 
while True:
    client, addr = s.accept()
    client.settimeout(5)
    while True:
        content = client.recv(1024)
        if len(content) ==0:
           break
        if str(content,'utf-8') == '\r\n':
            continue
        else:
            print(str(content,'utf-8'))
            client.send(b'Hello From Python')
    client.close()