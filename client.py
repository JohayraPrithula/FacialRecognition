import socket


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1235))
s.connect(("192.168.0.108", 1235))

while True:
    client, addr = s.accept()
    client.settimeout(5)
    content = client.recv(1)
    print(str(content,"utf-8"))
    client.close()