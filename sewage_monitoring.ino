/***************************************************************
  PROJECT TITLE  : Smart Sewage Monitoring and Alert System
  PLATFORM       : ESP32
  APPLICATION    : Real-Time Sewage Safety & Overflow Detection
***************************************************************/

#include <math.h>
#include <WiFi.h>
#include <PubSubClient.h>

// ---------------- WIFI CONFIGURATION ----------------
const char* ssid = "Airtel_Sundar";
const char* password = "Rakesh12531";

// ---------------- MQTT CONFIGURATION ----------------
const char* mqttServer = "broker.hivemq.com";
const int   mqttPort   = 1883;
const char* mqttTopic = "sewage/sensor/data";

WiFiClient espClient;
PubSubClient mqttClient(espClient);

// ---------------- SENSOR PIN CONFIGURATION ----------------
#define LM35_PIN   34
#define MQ135_PIN  35
#define MQ4_PIN    32
#define TRIG_PIN   5
#define ECHO_PIN   18
#define FLOW_PIN   25

// ---------------- ALERT DEVICE CONFIGURATION ----------------
#define LED_GREEN  12
#define LED_YELLOW 13
#define LED_RED    14
#define BUZZER     27

// ---------------- GAS SENSOR CALIBRATION CONSTANTS ----------------
float RLOAD      = 10.0;
float R0_MQ135   = 76.63;
float R0_MQ4     = 10.0;

// ---------------- FLOW SENSOR VARIABLES ----------------
volatile unsigned long pulseCount = 0;
unsigned long lastFlowTime = 0;
float totalLitres = 0.0;
float calibrationFactor = 7.5;

// ---------------- TANK PARAMETERS ----------------
#define TANK_HEIGHT 15.0  

// ---------------- FLOW INTERRUPT ----------------
void IRAM_ATTR pulseISR() {
  pulseCount++;
}

// ---------------- WIFI CONNECT ----------------
void connectWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi Connected");
}

// ---------------- MQTT CONNECT ----------------
void connectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("Connecting to MQTT...");
    if (mqttClient.connect("ESP32_Sewage_Node")) {
      Serial.println("✅ Connected");
    } else {
      Serial.println("❌ Failed. Retrying...");
      delay(2000);
    }
  }
}

// ---------------- LM35 ----------------
float readLM35() {
  long sum = 0;
  for (int i = 0; i < 50; i++) {
    sum += analogRead(LM35_PIN);
    delay(2);
  }
  float avgADC = sum / 50.0;
  float voltage = avgADC * (3.15 / 4095.0);
  return voltage * 100.0;
}

// ---------------- MQ ----------------
float readMQ_RS(int pin) {
  int raw = analogRead(pin);
  float vOut = raw * (3.3 / 4095.0);
  if (vOut < 0.01) vOut = 0.01;
  return (3.3 - vOut) * RLOAD / vOut;
}

float readMQIndex(int pin, float R0) {
  float RS = readMQ_RS(pin);
  float ratio = RS / R0;
  return 1000.0 * pow(ratio, -2.95);
}

// ---------------- ULTRASONIC ----------------
float readUltrasonicDistance() {
  digitalWrite(TRIG_PIN, LOW); delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1;
  return duration * 0.0343 / 2.0;
}

// ---------------- RISK LOGIC ----------------
int computeRisk(float tempC, float mq135Index, float mq4Index,
                float waterLevelPercent, float flowLpm) {

  int risk = 0;

  if (tempC > 40.0) risk = max(risk, 1);
  if (flowLpm > 40.0) risk = max(risk, 1);

  if (mq135Index > 300000.0) risk = 2;
  else if (mq135Index > 190000.0) risk = max(risk, 1);

  if (mq4Index > 1.8) risk = 2;
  else if (mq4Index > 1.8) risk = max(risk, 1);

  if (waterLevelPercent >= 70.0) risk = 2;
  else if (waterLevelPercent >= 40.0) risk = max(risk, 1);

  return risk;
}

// ---------------- ALERT OUTPUT ----------------
void applyAlerts(int riskLevel) {
  digitalWrite(LED_GREEN, riskLevel == 0);
  digitalWrite(LED_YELLOW, riskLevel == 1);
  digitalWrite(LED_RED, riskLevel == 2);
  digitalWrite(BUZZER, riskLevel == 2);
}

// ---------------- MQTT PUBLISH ----------------
void publishMQTT(float temp, float mq135, float mq4,
                 float waterLevel, float flow, int risk) {

  if (!mqttClient.connected()) connectMQTT();
  mqttClient.loop();

  String payload = "{";
  payload += "\"temp\":" + String(temp) + ",";
  payload += "\"mq135\":" + String(mq135) + ",";
  payload += "\"mq4\":" + String(mq4) + ",";
  payload += "\"waterLevel\":" + String(waterLevel) + ",";
  payload += "\"flow\":" + String(flow) + ",";
  payload += "\"risk\":" + String(risk);
  payload += "}";

  mqttClient.publish(mqttTopic, payload.c_str());
}

// ---------------- SETUP ----------------
void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(FLOW_PIN, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(FLOW_PIN), pulseISR, FALLING);

  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(BUZZER, OUTPUT);

  connectWiFi();
  mqttClient.setServer(mqttServer, mqttPort);
  connectMQTT();

  lastFlowTime = millis();
  Serial.println("✅ SYSTEM INITIALIZED WITH MQTT");
}

// ---------------- LOOP ----------------
void loop() {

  float tempC = readLM35();
  float mq135Index = readMQIndex(MQ135_PIN, R0_MQ135);
  float mq4Index   = readMQIndex(MQ4_PIN, R0_MQ4);
  float distance  = readUltrasonicDistance();

  float waterLevelPercent = ((TANK_HEIGHT - distance) / TANK_HEIGHT) * 100.0;
  waterLevelPercent = constrain(waterLevelPercent, 0, 100);

  if (millis() - lastFlowTime >= 1000) {

    detachInterrupt(digitalPinToInterrupt(FLOW_PIN));

    float freq = pulseCount;
    float flowLpm = freq / calibrationFactor;

    pulseCount = 0;
    lastFlowTime = millis();

    attachInterrupt(digitalPinToInterrupt(FLOW_PIN), pulseISR, FALLING);

    int risk = computeRisk(tempC, mq135Index, mq4Index,
                           waterLevelPercent, flowLpm);

    applyAlerts(risk);

    publishMQTT(tempC, mq135Index, mq4Index,
                waterLevelPercent, flowLpm, risk);

    Serial.println("✅ Data Sent to MQTT");
  }

  delay(50);
}   