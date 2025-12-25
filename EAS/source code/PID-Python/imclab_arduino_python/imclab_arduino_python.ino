/*
  iMCLab Internet-Based Motor Control Lab
  Fixed Version for RPM Reading
*/

#include <Arduino.h>

// --- Constants ---
const String vers = "1.0";     // version of this firmware
const int baud = 115200;       // serial baud rate
const char sp = ' ';           // command separator
const char nl = '\n';          // command terminator

// --- Pin Definitions (ESP32) ---
// Motor Driver L298N
const int motor1Pin1 = 27;     // Input 1 L298N
const int motor1Pin2 = 26;     // Input 2 L298N
const int enable1Pin = 12;     // Enable A L298N (PWM Speed)

// LED Built-in / External
const int pinLED  = 2;

// Sensor Speed (Encoder)
const byte pin_rpm = 13;       // Output Sensor masuk ke Pin 13

// --- PWM Properties ---
const int freq = 30000;
const int pwmChannel = 1;      // Channel untuk Motor
const int ledChannel = 0;      // Channel untuk LED
const int resolution = 8;      // 8-bit resolution (0-255)

// --- RPM Variables ---
volatile unsigned long rev = 0;        // Variable interrupt (Wajib volatile)
unsigned long last_rev_count = 0;      
unsigned long last_rpm_time = 0;       

float rpm = 0;
float rpm_filtered = 0;

// --- Global Variables ---
char Buffer[64];
String cmd;                    
double pv = 0;                 
float level;
double op = 0;                 
int iwrite = 0;
int n = 10;

// Fungsi ini dipanggil otomatis setiap sensor mendeteksi lubang/magnet
void IRAM_ATTR isr() {
  rev++;
}

// --- Parsing Serial (Sama seperti sebelumnya) ---
void parseSerial(void) {
  int ByteCount = Serial.readBytesUntil(nl,Buffer,sizeof(Buffer));
  String read_ = String(Buffer);
  memset(Buffer,0,sizeof(Buffer));
  
  int idx = read_.indexOf(sp);
  cmd = read_.substring(0,idx);
  cmd.trim();
  cmd.toUpperCase();

  String data = read_.substring(idx+1);
  data.trim();
  pv = data.toFloat();
}

// --- Menghitung RPM ---
void calculateRPM() {
  unsigned long current_time = millis();
  unsigned long time_elapsed = current_time - last_rpm_time;
  
  // Hitung RPM setiap 1000ms (1 detik) agar stabil
  if (time_elapsed >= 1000) { 
    noInterrupts(); // Stop interrupt sebentar agar pembacaan data aman
    unsigned long current_rev = rev;
    interrupts();   // Nyalakan lagi
    
    int holes = 2; // Ganti sesuai jumlah lubang di piringan encoder Anda
    float rotations = (float)(current_rev - last_rev_count) / holes;
    
    // Rumus: (Putaran / Waktu dalam menit)
    rpm = (rotations / (time_elapsed / 60000.0));
    
    // Low-pass filter sederhana
    rpm_filtered = 0.7 * rpm_filtered + 0.3 * rpm;
    
    last_rev_count = current_rev;
    last_rpm_time = current_time;
    
    // // Print RPM information
    // Serial.print("Time: ");
    // Serial.print(time_elapsed);
    // Serial.print("ms, Pulses: ");
    // Serial.print(current_rev - last_rev_count);
    // Serial.print(", RPM: ");
    // Serial.print(rpm);
    // Serial.print(", Filtered RPM: ");
    // Serial.println(rpm_filtered);
  }
}

// --- Eksekusi Perintah ---
void dispatchCommand(void) {
  if (cmd == "OP") {   
    // Mengatur Kecepatan Motor
    // Input pv dari Python 0-100%
    op = max(0.0, min(100.0, pv)); 
    
    // Mapping 0-100% ke 0-255 (PWM 8-bit)
    iwrite = map(op, 0, 100, 0, 255);
    
    ledcWrite(pwmChannel, iwrite);
    Serial.println(op);
  }
  else if (cmd == "RPM") {
    calculateRPM();    
    Serial.println(rpm); // Kirim nilai RPM ke Python
  }
  else if ((cmd == "V") or (cmd == "VER")) {
    Serial.println("iMCLab Firmware Version " + vers);
  }
  else if (cmd == "LED") {
    level = max(0.0, min(100.0, pv));
    iwrite = map(level, 0, 100, 0, 255);  
    ledcWrite(ledChannel, iwrite);      
    Serial.println(level);
  }  
  else if (cmd == "X") {
    ledcWrite(pwmChannel, 0);
    Serial.println("Stop");
  }
}

// --- 2. SETUP (Perbaikan Utama) ---
void setup() {
  Serial.begin(baud); 
  while (!Serial) { ; }

  // A. Konfigurasi L298N (Motor Driver)
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  // Set Arah Putaran (Maju)
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);

  // B. Konfigurasi PWM Motor (Speed)
  // PWM dipasang ke enable1Pin (Pin 12), BUKAN pin sensor
  ledcSetup(pwmChannel, freq, resolution);
  ledcAttachPin(enable1Pin, pwmChannel); 
  ledcWrite(pwmChannel, 0); // Pastikan motor mati saat start

  // C. Konfigurasi LED
  ledcSetup(ledChannel, freq, resolution);
  ledcAttachPin(pinLED, ledChannel);

  // D. Konfigurasi Sensor RPM (Encoder)
  // Wajib INPUT_PULLUP agar sinyal tidak floating
  pinMode(pin_rpm, INPUT_PULLUP); 
  // Pasang Interrupt: Panggil fungsi 'isr' setiap sinyal naik (RISING)
  attachInterrupt(digitalPinToInterrupt(pin_rpm), isr, RISING);
}

void loop() {
  parseSerial();
  dispatchCommand();
}