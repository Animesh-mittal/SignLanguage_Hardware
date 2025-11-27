#include "image_provider.h"
#include "Arduino.h"
#include "Arduino_OV767X.h"

// Define the camera capture resolution
#define CAM_CAPTURE_W 176
#define CAM_CAPTURE_H 144

// The two large buffers that were causing the memory issue
static unsigned short cam_color_buffer[CAM_CAPTURE_W * CAM_CAPTURE_H];
static uint8_t cam_grayscale_buffer[CAM_CAPTURE_W * CAM_CAPTURE_H];

// Converts a 16-bit RGB565 color frame to an 8-bit grayscale frame
void convertRGB565toGrayscale(const unsigned short* color_buf, uint8_t* gray_buf, int pixel_count) {
  for (int i = 0; i < pixel_count; i++) {
    uint16_t pixel = color_buf[i];
    uint8_t r = (pixel >> 11) & 0x1F;
    uint8_t g = (pixel >> 5) & 0x3F;
    uint8_t b = (pixel) & 0x1F;
    r = (r * 255) / 31;
    g = (g * 255) / 63;
    b = (b * 255) / 31;
    gray_buf[i] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
  }
}

// Resizes a large grayscale image to a small 32x32 image
static void resize_to_32x32(const uint8_t* src, uint8_t* dst) {
  const int srcW = CAM_CAPTURE_W;
  const int srcH = CAM_CAPTURE_H;
  const int outW = 32, outH = 32;

  for (int y = 0; y < outH; y++) {
    int src_y = (y * srcH) / outH;
    int src_y_offset = src_y * srcW;
    for (int x = 0; x < outW; x++) {
      int src_x = (x * srcW) / outW;
      dst[y * outW + x] = src[src_y_offset + src_x];
    }
  }
}

// Main function to get the image for the model
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter,
                      int wanted_width, int wanted_height,
                      int wanted_channels, uint8_t* image_data) {

  Camera.readFrame(cam_color_buffer);
  convertRGB565toGrayscale(cam_color_buffer, cam_grayscale_buffer, CAM_CAPTURE_W * CAM_CAPTURE_H);
  resize_to_32x32(cam_grayscale_buffer, image_data);

  return kTfLiteOk;
}