/*
 * AI-Powered Smart Safety Monitoring System
 * Industrial Workplace Safety and Hazard Detection
 * Arduino Uno Hardware Controller & MQ-2 Telemetry Source
 *
 * Hardware Pin Mapping:
 * - MQ-2 Analog Input:    A0 (gas sensor analog signal)
 * - Fan Relay Control:    Pin 7 (Active-LOW: LOW=ON, HIGH=OFF)
 * - Warning Buzzer:       Pin 8 (Active-HIGH)
 * - Warning Red LED:      Pin 9 (Active-HIGH)
 *
 * Serial Protocol:
 * - Baud rate: 9600
 * - Format: Newline-terminated JSON strings
 *   Schema: {"gas_raw":<int>,"gas_ppm":null,"gas_hazard":<bool>,"fan":<0/1>,"buzzer":<0/1>,"led":<0/1>}
 */

const int GAS_PIN = A0;
const int RELAY_PIN = 7;
const int BUZZER_PIN = 8;
const int LED_PIN = 9;

// Default provisional raw ADC threshold (0-1023)
int gasThreshold = 400;

// Active-LOW relay definition
const int RELAY_ON = LOW;
const int RELAY_OFF = HIGH;

// State tracking
int fanState = 0;
int buzzerState = 0;
int ledState = 0;
bool hazardActive = false;

// Non-blocking telemetry timing
unsigned long lastTelemetryTime = 0;
const unsigned long TELEMETRY_INTERVAL = 500; // Emit every 500 ms (2 Hz)

void sendTelemetry(int rawVal, bool hazard, int fan, int buz, int led) {
  Serial.print("{\"gas_raw\":");
  Serial.print(rawVal);
  Serial.print(",\"gas_ppm\":null");
  Serial.print(",\"gas_hazard\":");
  Serial.print(hazard ? "true" : "false");
  Serial.print(",\"fan\":");
  Serial.print(fan);
  Serial.print(",\"buzzer\":");
  Serial.print(buz);
  Serial.print(",\"led\":");
  Serial.print(led);
  Serial.println("}");
}

void setup() {
  // Safe initial relay setup: pull HIGH before setting as OUTPUT to avoid boot glitch
  digitalWrite(RELAY_PIN, RELAY_OFF);
  pinMode(RELAY_PIN, OUTPUT);

  pinMode(GAS_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  digitalWrite(RELAY_PIN, RELAY_OFF);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(9600);
}

void loop() {
  // Read MQ-2 sensor raw ADC (0-1023)
  int gasRaw = analogRead(GAS_PIN);

  // Automatic gas threshold evaluation
  hazardActive = (gasRaw >= gasThreshold);

  if (hazardActive) {
    // Hazard state: activate fan relay, siren buzzer, red LED
    digitalWrite(RELAY_PIN, RELAY_ON);
    digitalWrite(BUZZER_PIN, HIGH);
    digitalWrite(LED_PIN, HIGH);
    fanState = 1;
    buzzerState = 1;
    ledState = 1;
  } else {
    // Normal state: de-activate all outputs
    digitalWrite(RELAY_PIN, RELAY_OFF);
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(LED_PIN, LOW);
    fanState = 0;
    buzzerState = 0;
    ledState = 0;
  }

  // Periodic telemetry emission
  unsigned long now = millis();
  if (now - lastTelemetryTime >= TELEMETRY_INTERVAL) {
    lastTelemetryTime = now;
    sendTelemetry(gasRaw, hazardActive, fanState, buzzerState, ledState);
  }
}
