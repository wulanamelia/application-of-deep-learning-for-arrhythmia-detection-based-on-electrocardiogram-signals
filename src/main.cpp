#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <SPI.h>
#include <SD.h>

// TensorFlow Lite Micro
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// SD Card
#define SD_CS 5  // Sesuaikan dengan pin CS pada SD card reader

// EKG sensor AD8232
#define EKG_PIN 36
#define LO_PLUS 32
#define LO_MINUS 33

// TensorFlow Lite
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// SD card file
File dataFile;

void setup() {
  Serial.begin(115200);
  pinMode(LO_PLUS, INPUT);
  pinMode(LO_MINUS, INPUT);

  // OLED init
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED gagal");
    while (true);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  // SD Card init
  if (!SD.begin(SD_CS)) {
    Serial.println("SD gagal");
    display.println("SD gagal");
    display.display();
    while (true);
  }
  dataFile = SD.open("/log.txt", FILE_WRITE);

  // TensorFlow init
  const tflite::Model* model = tflite::GetModel(model_tflite);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);

  display.println("Sistem Siap");
  display.display();
}

void loop() {
  if (digitalRead(LO_PLUS) == 1 || digitalRead(LO_MINUS) == 1) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("Elektroda lepas!");
    display.display();
    delay(1000);
    return;
  }

  int ekg_raw = analogRead(EKG_PIN);
  float ekg_input = (float)ekg_raw / 4095.0;

  input->data.f[0] = ekg_input;

  interpreter->Invoke();
  float result = output->data.f[0];

  String status = "Unknown";
  if (result < 0.33) status = "Normal";
  else if (result < 0.66) status = "Bradikardia";
  else status = "Takikardia";

  display.clearDisplay();
  display.setCursor(0, 0);
  display.print("EKG: ");
  display.println(ekg_raw);
  display.print("Status: ");
  display.println(status);
  display.display();

  if (dataFile) {
    dataFile.print("EKG:"); dataFile.print(ekg_raw);
    dataFile.print(", Status:"); dataFile.println(status);
    dataFile.flush();
  }

  delay(500);
}
