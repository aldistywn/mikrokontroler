#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "OFFICE KEDAI 38";
const char* password =  "12345678";

const char* mqttServer = "broker.emqx.io";
const int mqttPort = 1883;

WiFiClient espClient;
PubSubClient client(espClient);

int ledPin = 2;
int motor1Pin1 = 27;
int motor1Pin2 = 26;
int enable1Pin = 12;

const int freq = 30000;
const int pwmChannel = 0;
const int resolution = 8;

const byte pin_rpm = 13;
volatile int rev = 0;
float rpm = 0;
unsigned long lastMillis = 0;

void IRAM_ATTR isr() {
  rev++;
}

void receivedCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  if (String(topic) == "kontrol") {
    int speed = message.toInt();
    speed = constrain(speed, 0, 200);

    if (speed > 0) {
      digitalWrite(motor1Pin1, HIGH);
      digitalWrite(motor1Pin2, LOW);
      ledcWrite(pwmChannel, speed);
      digitalWrite(ledPin, HIGH);
    } else {
      digitalWrite(motor1Pin1, LOW);
      digitalWrite(motor1Pin2, LOW);
      ledcWrite(pwmChannel, 0);
      digitalWrite(ledPin, LOW);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(ledPin, OUTPUT);

  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enable1Pin, OUTPUT);
  
  pinMode(pin_rpm, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(pin_rpm), isr, RISING);

  ledcSetup(pwmChannel, freq, resolution);
  ledcAttachPin(enable1Pin, pwmChannel);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }

  client.setServer(mqttServer, mqttPort);
  client.setCallback(receivedCallback);

  while (!client.connected()) {
    if (client.connect("IMCLab_Motor_Client")) {
      client.subscribe("kontrol");
    } else {
      delay(2000);
    }
  }
}

void loop() {
  if (!client.connected()) {
    // Reconnect logic can be added here
  }
  client.loop();

  if (millis() - lastMillis >= 1000) {
    detachInterrupt(digitalPinToInterrupt(pin_rpm));
    
    const float pulses_per_rev = 20.0; 
    rpm = (rev / pulses_per_rev) * 60.0;
    
    char rpmString[8];
    dtostrf(rpm, 1, 2, rpmString);
    client.publish("rpm", rpmString);

    rev = 0;
    lastMillis = millis();
    attachInterrupt(digitalPinToInterrupt(pin_rpm), isr, RISING);
  }
}