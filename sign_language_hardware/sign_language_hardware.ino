#include <Arduino.h>
#include <TensorFlowLite.h>
#include <Arduino_OV767X.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "sign_model.h" 
#include "image_provider.h"

using namespace tflite;

// Globals for TensorFlow
ErrorReporter* error_reporter;
AllOpsResolver resolver;
MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// The Tensor Arena
constexpr int kTensorArenaSize = 25 * 1024; 
static uint8_t tensor_arena[kTensorArenaSize];

// CHANGED: Labels updated for your 5 classes
const char* kLabels[5] = {"A", "B", "C", "Nothing", "Space"};

// Helper function to clamp values
static inline int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Quantize the image to the model's input format
void quantize_to_model_input(const uint8_t* gray_32x32, TfLiteTensor* in) {
  const float s = in->params.scale;
  const int zp  = in->params.zero_point;

  if (in->type == kTfLiteInt8) {
    for (int i = 0; i < 32 * 32; i++) {
      float f = gray_32x32[i] / 255.0f;
      int q = lroundf(f / s) + zp;
      in->data.int8[i] = (int8_t)clamp_int(q, -128, 127);
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  static MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  // Initialize camera in the compatible RGB565 mode
  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("Camera begin failed!");
    while (1);
  }
  Serial.println("Camera ready.");

  // Map model
  // NOTE: Make sure the C array in your sign_model.h file is named 'model'
  const Model* tflModel = GetModel(model); 
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1) {}
  }

  static MicroInterpreter static_interpreter(tflModel, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (1) {}
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("Setup complete.");
}

void loop() {
  // Buffer to hold the final 32x32 image
  static uint8_t image_data[32 * 32];

  // Get the image from the camera
  if (GetImage(error_reporter, 32, 32, 1, image_data) != kTfLiteOk) {
    error_reporter->Report("Image capture failed.");
    return;
  }

  // Quantize the image for the model
  quantize_to_model_input(image_data, input);

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed.");
    return;
  }

  // Get dequantization parameters from the output tensor
  const float output_scale = output->params.scale;
  const int output_zero_point = output->params.zero_point;

  // Find the highest score to determine the predicted letter
  int8_t best_score = -128;
  int best_index = 0;
  for (int i = 0; i < 5; i++) { // CHANGED: Loop for 5 classes
    if (output->data.int8[i] > best_score) {
      best_score = output->data.int8[i];
      best_index = i;
    }
  }

  // --- SINGLE-LINE PRINTING LOGIC ---
  Serial.print("Prediction: ");
  Serial.print(kLabels[best_index]);
  Serial.print(" | Scores: ");

  // Loop through all classes to print their scores on the same line
  for (int i = 0; i < 5; i++) { // CHANGED: Loop for 5 classes
    int8_t raw_score = output->data.int8[i];
    float probability = (float)(raw_score - output_zero_point) * output_scale;

    Serial.print(kLabels[i]);
    Serial.print(": ");
    Serial.print(probability * 100, 1);
    Serial.print("%");

    // Add a separator, but not after the last item
    if (i < 4) { // CHANGED: Condition for 5 classes
      Serial.print(" | ");
    }
  }

  // End the complete line of data
  Serial.println();
  // Add the extra blank line for spacing
  Serial.println(); 

  delay(1500);
}