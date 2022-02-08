
#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// // Update these with values suitable for the network.alues suitable for the network.

const char* ssid = "Urbita";
const char* password = "19p16u14n";
const char* mqtt_server = "broker.mqttdashboard.com";

// Values suitable for the motor.

#define STEPPER_PIN_1 D5
#define STEPPER_PIN_2 D6
#define STEPPER_PIN_3 D7
#define STEPPER_PIN_4 D8
int step_number = 0;


// Values suitable for the door.
int doorState = 0;
int b1 = 0;
int b2 = 0;
int ocmd = 0;
int zero1 = 0;
int zero2 = 0;

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {

  // We start by connecting to a WiFi network
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
}



void callback(char* topic, byte* payload, unsigned int length) 
{
  String c;
  for (int i = 0; i < length; i++) 
    c += (char)payload[i];
  if (c == "1")
    ocmd = 1;
}



void reconnect() 
{
  // Loop until we're reconnected
  while (!client.connected()) 
  {
    Serial.print("Attempting MQTT connection...");
    // Create a random client ID
    String clientId = "ESP8266Client-";
    clientId += String(random(0xffff), HEX);
    // Attempt to connect
    if (client.connect(clientId.c_str(),ssid, password))
    {
      Serial.println("connected");
      client.subscribe("openCommand");
    }
    else 
    {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}



void setup() 
{
  
  Serial.begin(115200);
  pinMode(D3,INPUT_PULLUP);
  pinMode(D4,INPUT_PULLUP);
  pinMode(STEPPER_PIN_1, OUTPUT);
  pinMode(STEPPER_PIN_2, OUTPUT);
  pinMode(STEPPER_PIN_3, OUTPUT);
  pinMode(STEPPER_PIN_4, OUTPUT);
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}



void loop() 
{
  int button1=digitalRead(D3);
  int button2=digitalRead(D4);
  
  if (!client.connected()) 
    reconnect();
  client.loop();

 // Publishing data to doorbell

  if (zero1 >= 50 && button1 == 1)
  {
    b1 = 1;
    Serial.println();
    Serial.println();
    Serial.print("button1: ");
    Serial.println(zero1);
    if (doorState == 0)
    {
      Serial.println("Publishing data..");
      client.publish("DoorBell1", "1");
      b1 = 0;
    }
  }
  if (button1 == 1)
    zero1 = 0;
  if (button1 == 0)
    zero1 += 1; 


  if (zero2 >= 50 && button2 == 1)
  {
    b2 = 1;
    Serial.println();
    Serial.println();
    Serial.print("button2: ");
    Serial.println(zero2);
    client.publish("DoorBell1", "1");
    
  }
  if (button2 == 1)
    zero2 = 0;
  if (button2 == 0)
    zero2 += 1;

  if (doorState ==0 && ((b2 == 1) || (ocmd == 1)))
  {
    doorOpen();
    doorState = stateToggle(doorState);
    b2 = 0;
    ocmd = 0;
    Serial.print("..opened");
  }
  if (doorState == 1 && ((b1 == 1) || (b2 == 1)))
  {
    doorClose();
    doorState = stateToggle(doorState);
    b1 = 0;
    b2 = 0;
    Serial.print("..closed");
  }  
}


int stateToggle(int prev)
{
  if (prev == 1)
    return 0;
  else
    return 1;
}


void doorOpen()
{
  Serial.print("Door Opening");
  for (int a = 0; a<750; a++)
  {
    switch(step_number)
    {
      case 0:
      digitalWrite(STEPPER_PIN_1, HIGH);
      digitalWrite(STEPPER_PIN_2, LOW);
      digitalWrite(STEPPER_PIN_3, LOW);
      digitalWrite(STEPPER_PIN_4, LOW);
      break;
      case 1:
      digitalWrite(STEPPER_PIN_1, LOW);
      digitalWrite(STEPPER_PIN_2, HIGH);
      digitalWrite(STEPPER_PIN_3, LOW);
      digitalWrite(STEPPER_PIN_4, LOW);
      break;
      case 2:
      digitalWrite(STEPPER_PIN_1, LOW);
      digitalWrite(STEPPER_PIN_2, LOW);
      digitalWrite(STEPPER_PIN_3, HIGH);
      digitalWrite(STEPPER_PIN_4, LOW);
      break;
      case 3:
      digitalWrite(STEPPER_PIN_1, LOW);
      digitalWrite(STEPPER_PIN_2, LOW);
      digitalWrite(STEPPER_PIN_3, LOW);
      digitalWrite(STEPPER_PIN_4, HIGH);
      break;
    } 
    step_number++;
    if(step_number > 3)
    {
      step_number = 0;
    }
    delay(5);
  }
  step_number = 0;
  digitalWrite(STEPPER_PIN_1, LOW);
  digitalWrite(STEPPER_PIN_2, LOW);
  digitalWrite(STEPPER_PIN_3, LOW);
  digitalWrite(STEPPER_PIN_4, LOW);
}

void doorClose()
{
  Serial.print("Door Closing");
  for (int a = 0; a<750; a++)
  {
    switch(step_number)
    {
      case 0:
      digitalWrite(STEPPER_PIN_1, LOW);
      digitalWrite(STEPPER_PIN_2, LOW);
      digitalWrite(STEPPER_PIN_3, LOW);
      digitalWrite(STEPPER_PIN_4, HIGH);
      break;
      case 1:
      digitalWrite(STEPPER_PIN_1, LOW);
      digitalWrite(STEPPER_PIN_2, LOW);
      digitalWrite(STEPPER_PIN_3, HIGH);
      digitalWrite(STEPPER_PIN_4, LOW);
      break;
      case 2:
      digitalWrite(STEPPER_PIN_1, LOW);
      digitalWrite(STEPPER_PIN_2, HIGH);
      digitalWrite(STEPPER_PIN_3, LOW);
      digitalWrite(STEPPER_PIN_4, LOW);
      break;
      case 3:
      digitalWrite(STEPPER_PIN_1, HIGH);
      digitalWrite(STEPPER_PIN_2, LOW);
      digitalWrite(STEPPER_PIN_3, LOW);
      digitalWrite(STEPPER_PIN_4, LOW);
    } 
    step_number++;
    if(step_number > 3)
    {
      step_number = 0;
    }
    delay(5);
  } 
  step_number = 0;
  digitalWrite(STEPPER_PIN_1, LOW);
  digitalWrite(STEPPER_PIN_2, LOW);
  digitalWrite(STEPPER_PIN_3, LOW);
  digitalWrite(STEPPER_PIN_4, LOW);
}
