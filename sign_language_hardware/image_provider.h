#ifndef IMAGE_PROVIDER_H_
#define IMAGE_PROVIDER_H_

#include <TensorFlowLite.h>  
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/c/common.h" 

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter,
                      int wanted_width, int wanted_height, int wanted_channels,
                      uint8_t* image_data);

#endif  // IMAGE_PROVIDER_H_
