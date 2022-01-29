#include <Arduino.h>
#include <ESP8266WiFi.h>
#include <WiFiClient.h>

const uint16_t port = 1235;
const char *host = "192.168.0.108";
WiFiClient client;
int one = 0;
int zero = 0;
void setup()
{
    pinMode(D3,INPUT_PULLUP);
    Serial.begin(115200);
    Serial.println("Connecting...\n");
    WiFi.mode(WIFI_STA);
    WiFi.begin("Urbita", "19p16u14n"); // change it to your ussid and password
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());

}

void loop()
{
    int button=digitalRead(D3);
    
    
    if (!client.connect(host, port))
    {
        Serial.println("Connection to host failed");
        delay(1000);
        return;
    }

    client.println("2");
    
    if (zero >= 5 && button == 1)
      {client.println("1");
      Serial.println(zero);}
    if (button == 1)
      zero = 0;
    if (button == 0)
      zero += 1;
    
    
    
    
    client.stop();
    delay(2);
}
