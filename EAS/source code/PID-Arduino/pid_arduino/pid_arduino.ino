/*
  iMCLab Internet-Based Motor Control Lab
  Kit internet-Based Motor Control Lab (iMCLab), adalah modul
  hardware kendali motor DC umpan balik dengan mikrokontroller
  ESP32, yang terdiri dari motor DC, driver motor, LED, dan
  sensor kecepatan. Kit ini dikembangkan oleh Kampus Bela
  Negara. Keluaran kecepatan motor DC dalam rotasi per menit
  (rpm) disesuaikan untuk mempertahankan setpoint rpm yang
  diinginkan. Kit ini dikembangkan oleh tim kami, sebagai
  salah satu luaran tambahan dari Penelitian Terapan Unggulan
  Perguruan Tinggi (PTUPT) Tahun Kedua.
*/

// constants
const int baud = 115200;       // serial baud rate

int motor1Pin1 = 27;
int motor1Pin2 = 26;
int enable1Pin = 12;

// Setting PWM properties
const int freq = 30000;
const int pwmChannel = 1;
const int resolution = 8;

const byte pin_rpm = 13;
volatile unsigned long rev = 0;        // Changed to unsigned long
unsigned long last_rev_count = 0;      // Track last revolution count
unsigned long last_rpm_time = 0;       // Track last RPM calculation time

float rpm = 0;                         // Current RPM value
float rpm_filtered = 0;                // Filtered RPM value

unsigned long cur_time, old_time;

//float P, I, D, Kc, tauI, tauD;
// ===============================================
// PARAMETER PID AWAL UNTUK TUNING
// ===============================================
float P, I, D;
float Kc = 0.001;  // MULAI DENGAN GAIN YANG SANGAT KECIL
//float tauI = 9999.0; // Nilai besar untuk mematikan Integral
//float tauD = 0.0;    // Nilai 0 untuk mematikan Derivatif
float tauI = 1; // Nilai besar untuk mematikan Integral
float tauD = 1;    // Nilai 0 untuk mematikan Derivatif
// ===============================================

float KP, KI, KD, op0, ophi, oplo, error, dpv;
float sp = 5000,                        // set point
      pv = 0,                          // current RPM
      pv_last = 0,                     // prior RPM
      ierr = 0,                        // integral error
      dt = 0;                          // time between measurements
int op = 0;                            // PID controller output (changed to int for PWM)
unsigned long ts = 0, new_ts = 0;      // timestamp

void IRAM_ATTR isr() {
  rev++;
}

void setup() {
  ts = millis();
  
  Serial.begin(baud); 
  while (!Serial) {
    ; // wait for serial port to connect.
  }
  
  // sets the pins
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enable1Pin, OUTPUT);
  pinMode(pin_rpm, INPUT_PULLUP);
  
  attachInterrupt(digitalPinToInterrupt(pin_rpm), isr, RISING);

  // configure LED PWM functionalitites
  ledcSetup(pwmChannel, freq, resolution);

  // attach the channel to the GPIO to be controlled
  ledcAttachPin(enable1Pin, pwmChannel);

  // Initialize motor direction
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);

  // testing
  Serial.println("Testing DC Motor...");
  last_rpm_time = millis();
}

void calculateRPM() {
  unsigned long current_time = millis();
  unsigned long time_elapsed = current_time - last_rpm_time;
  
  if (time_elapsed >= 1000) { // Calculate RPM every second
    noInterrupts(); // Disable interrupts briefly for safe reading
    unsigned long current_rev = rev;
    interrupts(); // Re-enable interrupts
    
    // Calculate RPM
    int holes = 2; // holes of rotating object
    float rotations = (float)(current_rev - last_rev_count) / holes;
    rpm = (rotations * 60000.0) / time_elapsed; // Convert to RPM
    
    // Simple low-pass filter for RPM
    rpm_filtered = 0.7 * rpm_filtered + 0.3 * rpm;
    
    last_rev_count = current_rev;
    last_rpm_time = current_time;
    
    // Print RPM information
    Serial.print("Time: ");
    Serial.print(time_elapsed);
    Serial.print("ms, Pulses: ");
    Serial.print(current_rev - last_rev_count);
    Serial.print(", RPM: ");
    Serial.print(rpm);
    Serial.print(", Filtered RPM: ");
    Serial.println(rpm_filtered);
  }
}

float pid(float sp, float pv, float pv_last, float& ierr, float dt) {
//  float Kc = 10.0; // K / %Heater
//  float tauI = 3.0; // sec
//  float tauD = 1.0;  // sec
  
  // PID coefficients
  float KP = Kc;
  float KI = Kc / tauI;
  float KD = Kc * tauD; 
  
  // upper and lower bounds on output (PWM value)
  float ophi = 255;  // Maximum PWM value
  float oplo = 0;    // Minimum PWM value
  
  // calculate the error
  float error = sp - pv;
  
  // calculate the integral error
  ierr = ierr + KI * error * dt;  
  
  // calculate the measurement derivative
  float dpv = (pv - pv_last) / dt;
  
  // calculate the PID output
  float P = KP * error; // proportional contribution
  float I = ierr; // integral contribution
  float D = -KD * dpv; // derivative contribution
  float op = P + I + D;
  
  // implement anti-reset windup
  if ((op < oplo) || (op > ophi)) {
    ierr = ierr - KI * error * dt;
    // clip output
    op = max(oplo, min(ophi, op));
  }
  
  Serial.println("sp=" + String(sp) + " pv=" + String(pv) + " dt=" + String(dt) + 
                 " op=" + String(op) + " P=" + String(P) + " I=" + String(I) + " D=" + String(D));
  return op;
}

void loop() {
  new_ts = millis();
  
  // Calculate RPM continuously
  calculateRPM();
  
  if (new_ts - ts >= 1000) { // Run PID control every second
    //pv = rpm_filtered;   // Use filtered RPM for PID control
    pv = rpm;   // Use RPM for PID control
    dt = (new_ts - ts) / 1000.0;
    ts = new_ts;
    
    op = pid(sp, pv, pv_last, ierr, dt);
    
    // Ensure PWM value is within valid range
    // op = constrain(op, 0, 255);
    ledcWrite(pwmChannel, op);

    pv_last = pv;
    
    Serial.print("PWM: ");
    Serial.println(op);
  }
  
  delay(10); // Small delay to prevent overwhelming the serial port
}