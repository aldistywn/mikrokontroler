int ledPin = 2;
int motor1Pin1 = 27;
int motor1Pin2 = 26;
int enable1Pin = 12;

const int freq = 30000;
const int pwmChannel = 0;
const int resolution = 8;

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enable1Pin, OUTPUT);

  Serial.begin(115200);

  ledcSetup(pwmChannel, freq, resolution);

  ledcAttachPin(enable1Pin, pwmChannel);

  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);

  Serial.println("Arduino siap menerima data kecepatan motor...");
}

void loop() {
  if (Serial.available() > 0) {

    int speed = Serial.read();

    ledcWrite(pwmChannel, speed);

    if (speed > 0) {
      digitalWrite(ledPin, HIGH);
    } else {
      digitalWrite(ledPin, LOW);
    }

    Serial.print("Kecepatan motor diatur ke: ");
    Serial.println(speed);
  }
}