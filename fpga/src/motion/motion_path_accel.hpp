#ifndef MOTION_PATH_ACCEL_HPP
#define MOTION_PATH_ACCEL_HPP
#include <stdint.h>

// Non-visual, sampled forward kinematics only. No actuator interface.
static const unsigned MOTION_MAX_SEGMENTS = 64;
static const unsigned MOTION_INPUT_WORDS = 12;
static const unsigned MOTION_OUTPUT_WORDS = 8;
static const uint32_t MOTION_MAGIC = 0x4d504131; // MPA1
static const int MOTION_SAMPLES = 101;
static const int64_t MOTION_ERROR_Q24 = 32768; // 1/512 mm, outward bound

extern "C" void motion_path_accel(const uint32_t *input_words,
                                  uint32_t *output_words,
                                  unsigned segment_count, uint32_t request_tag);
#endif
