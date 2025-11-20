// Arduino sketch - envia ECG, IR, RED y ADXL345 en una línea por ciclo
// Usa la librería SparkFun MAX3010x (MAX30105.h) que ya tenías.

#include <Wire.h>
#include "MAX30105.h"

#define ADXL345_ADDR 0x53

// AD8232
const int ECG_PIN = 34;
const int LOP_PIN = 32;
const int LON_PIN = 33;
const bool LEADOFF_ACTIVO_ALTO = true;

// MAX30102
MAX30105 particleSensor;

// timing
const unsigned long PERIOD_MS = 50; // 20 Hz aprox (ajusta si quieres más lento/rápido)

void setup() {
  Serial.begin(115200);
  delay(300);

  pinMode(LOP_PIN, INPUT);
  pinMode(LON_PIN, INPUT);

  Wire.begin(21, 22); // SDA, SCL

  // Inicializar MAX30102
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("ERROR: MAX30102 no detectado");
  } else {
    // setup estándar; aumentamos amplitud de LED para conseguir señal más fuerte
    particleSensor.setup();  // configuración por defecto
    particleSensor.setPulseAmplitudeRed(0x3F); // máxima potencia LED rojo
    particleSensor.setPulseAmplitudeIR(0x3F);  // máxima potencia LED IR
    // Opcional: si tu librería soporta setSampleRate o setPulseWidth, ajusta aquí
  }

  // Inicializar ADXL345 (modo medida)
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(0x2D); // POWER_CTL
  Wire.write(0x08); // Measure = 1
  Wire.endTransmission();

  Serial.println("READY");
}

void loop() {
  unsigned long t0 = millis();

  // --- ECG AD8232 ---
  int lop = digitalRead(LOP_PIN);
  int lon = digitalRead(LON_PIN);
  bool electrodo_suelto = false;
  if (LEADOFF_ACTIVO_ALTO) {
    if (lop == HIGH || lon == HIGH) electrodo_suelto = true;
  }
  int ecg = electrodo_suelto ? 0 : analogRead(ECG_PIN);

  // --- MAX30102: leer IR y RED (crudos) ---
  long irValue = 0;
  long redValue = 0;

  // try to get values; getIR/getRed retorna valor actual según tu librería
  // si tu librería tiene FIFO/available(), podrías leer en bloque; aquí usamos getIR/getRed
  irValue = particleSensor.getIR();
  redValue = particleSensor.getRed();

  // --- ADXL345: leer X,Y,Z (6 bytes desde 0x32) ---
  int16_t x = 0, y = 0, z = 0;
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(0x32);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR, 6, true);
  if (Wire.available() == 6) {
    x = (int16_t)(Wire.read() | (Wire.read() << 8));
    y = (int16_t)(Wire.read() | (Wire.read() << 8));
    z = (int16_t)(Wire.read() | (Wire.read() << 8));
  }

  // --- Imprimir en formato único (Python espera IR/RED crudos)
  // ECG:### | IR:##### | RED:##### | AX:### | AY:### | AZ:###
  Serial.print("ECG:");
  Serial.print(ecg);
  Serial.print(" | IR:");
  Serial.print(irValue);
  Serial.print(" | RED:");
  Serial.print(redValue);
  Serial.print(" | AX:");
  Serial.print(x);
  Serial.print(" | AY:");
  Serial.print(y);
  Serial.print(" | AZ:");
  Serial.println(z);

  // Mantén la frecuencia de muestreo razonable
  unsigned long dt = millis() - t0;
  if (dt < PERIOD_MS) delay(PERIOD_MS - dt);
}

