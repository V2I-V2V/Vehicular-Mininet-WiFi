import socket
import time

def netcat(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, int(port)))
    # s.shutdown(socket.SHUT_WR)
    while True:
        data = s.recv(4096)
        if not data:
            break
        line = data.decode('utf-8')
        # print(line)
        split = line.split(',')
        if line.startswith('$GPGGA'):
            longitude, latitude = float(split[2])/100, float(split[4])/100
            print('location', longitude, latitude, time.time())
        elif line.startswith('$GPRMC') and split[2] == 'A':
            longitude, latitude = float(split[3])/100, float(split[5])/100
            print('location', longitude, latitude, time.time())
        
    s.close()
    
if __name__ == '__main__':
    netcat('localhost', 20175)