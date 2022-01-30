
import time
import paho.mqtt.client as paho


doorBell = 0

broker="broker.mqttdashboard.com"
#broker="iot.eclipse.org"
#define callback
flag_connected = 0

def on_connect(client, userdata, flags, rc):
   global flag_connected
   print("Connected")
   flag_connected = 1

def on_disconnect(client, userdata, rc):
   global flag_connected
   print("Disconnected")
   flag_connected = 0


def on_message(client, userdata, message):
    time.sleep(1)
    global doorBell
    rcv = (str(message.payload.decode("utf-8")))
    if rcv == "1":
        doorBell = 1
        print(doorBell)
    print("received message", rcv)

client= paho.Client("recognizerClient") #create client object reecognizerClient.on_publish = on_publish #assign function to callback reecognizerClient.connect(broker,port) #establish connection reecognizerClient.publish("house/bulb1","on")
######Bind function to callback
client.on_message=on_message
client.on_connect=on_connect
client.on_disconnect=on_disconnect
#####
print("connecting to broker ",broker)
client.connect(broker)#connect

while True:

    client.loop_start() #start loop to process received messages
    client.subscribe("DoorBell1")#subscribe
    time.sleep(1)
    #print("publishing ")
    print("outside",doorBell)
    if doorBell == 1:
        client.publish("openCommand","1")
        doorBell = 0
        time.sleep(1)
        
    client.loop_stop() #stop loop


client.disconnect() #disconnect