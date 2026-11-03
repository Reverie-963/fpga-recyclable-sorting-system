#include "motion_path_accel.hpp"
#include "motion_sine_q24.hpp"
#include <hls_stream.h>

struct Segment {
    int start[3];
    int end[3];
    uint32_t valid;
    uint32_t cost;
};
struct Result { uint32_t words[MOTION_OUTPUT_WORDS]; };

static unsigned checked_count(unsigned n) {
    return n >= 1 && n <= MOTION_MAX_SEGMENTS ? n : 0;
}

static void load_input(const uint32_t *input, unsigned n, hls::stream<Segment> &out) {
    for (unsigned row = 0; row < checked_count(n); ++row) {
        uint32_t words[MOTION_INPUT_WORDS];
        for (unsigned j = 0; j < MOTION_INPUT_WORDS; ++j) {
#pragma HLS PIPELINE II=1
            words[j] = input[row*MOTION_INPUT_WORDS+j];
        }
        Segment s;
        s.valid = 1;
        s.cost = 0;
        for (unsigned j = 0; j < 6; ++j) {
            if (words[j] < 500 || words[j] > 2500 ||
                words[j+6] < 500 || words[j+6] > 2500) s.valid = 0;
            // Invalid rows never participate in geometry or cost arithmetic.
            if (words[j] >= 500 && words[j] <= 2500 &&
                words[j+6] >= 500 && words[j+6] <= 2500) {
                int delta = int(words[j+6])-int(words[j]);
                s.cost += delta < 0 ? -delta : delta;
            }
        }
        for (unsigned j = 0; j < 3; ++j) {
            s.start[j] = s.valid ? int(words[j+1]) : 1500;
            s.end[j] = s.valid ? int(words[j+7]) : 1500;
        }
        out.write(s);
    }
}

static int32_t sine_node(unsigned index) {
#pragma HLS INLINE
    unsigned quadrant = (index >> 10) & 3;
    unsigned offset = index & 1023;
    int32_t value = MOTION_SINE_Q24[(quadrant & 1) ? 1024-offset : offset];
    return (quadrant & 2) ? -value : value;
}

// phase/800000 is a turn. Integer PWM interpolation makes phase exact;
// no floating-point angle reduction or trigonometric units are required.
static int32_t sine_phase(int phase) {
#pragma HLS INLINE
    int normalized = phase % 800000;
    if (normalized < 0) normalized += 800000;
    int64_t scaled = int64_t(normalized)*4096;
    unsigned index = unsigned(scaled/800000);
    int64_t remainder = scaled-int64_t(index)*800000;
    int32_t a = sine_node(index), b = sine_node(index+1);
    return a+int32_t((int64_t(b-a)*remainder)/800000);
}

static int64_t floor_q16(int64_t q24) {
#pragma HLS INLINE
    return q24 >= 0 ? q24/256 : -((-q24+255)/256);
}

static void compute(hls::stream<Segment> &in, unsigned n, uint32_t tag,
                    hls::stream<Result> &out) {
    for (unsigned row = 0; row < checked_count(n); ++row) {
        Segment s = in.read();
        int64_t minimum = INT64_C(1000000000000);
        unsigned argmin = 0;
        for (int i = 0; i < MOTION_SAMPLES; ++i) {
#pragma HLS PIPELINE II=3
            int p1 = (s.start[0]-1500)*100+(s.end[0]-s.start[0])*i;
            int p2 = (s.start[1]-1500)*100+(s.end[1]-s.start[1])*i;
            int p3 = (s.start[2]-1500)*100+(s.end[2]-s.start[2])*i;
            int a = 200000+3*p1, b = a-3*p2, c = b+3*p3;
            int64_t z = INT64_C(100)*16777216 +
                INT64_C(105)*sine_phase(a)+INT64_C(88)*sine_phase(b)+
                INT64_C(155)*sine_phase(c);
            if (z < minimum) { minimum = z; argmin = unsigned(i); }
        }
        Result result;
        result.words[0] = MOTION_MAGIC;
        result.words[1] = s.valid ? 0 : 1;
        result.words[2] = s.valid ? uint32_t(int32_t(floor_q16(minimum-MOTION_ERROR_Q24))) : 0x80000000;
        result.words[3] = s.valid ? uint32_t(int32_t(-floor_q16(-minimum-MOTION_ERROR_Q24))) : 0x7fffffff;
        result.words[4] = s.valid ? argmin : 0xffffffff;
        result.words[5] = s.valid ? MOTION_SAMPLES : 0;
        result.words[6] = s.valid ? s.cost : 0;
        result.words[7] = tag;
        out.write(result);
    }
}

static void store_output(hls::stream<Result> &in, unsigned n, uint32_t *output) {
    for (unsigned row = 0; row < checked_count(n); ++row) {
        Result r = in.read();
        for (unsigned j = 0; j < MOTION_OUTPUT_WORDS; ++j) {
#pragma HLS PIPELINE II=1
            output[row*MOTION_OUTPUT_WORDS+j] = r.words[j];
        }
    }
}

extern "C" void motion_path_accel(const uint32_t *input_words,
                                  uint32_t *output_words,
                                  unsigned segment_count, uint32_t request_tag) {
#pragma HLS INTERFACE m_axi port=input_words offset=slave bundle=input_mem depth=768 max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=output_words offset=slave bundle=output_mem depth=512 max_write_burst_length=16
#pragma HLS INTERFACE s_axilite port=input_words bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=output_words bundle=control offset=0x1c
#pragma HLS INTERFACE s_axilite port=segment_count bundle=control offset=0x28
#pragma HLS INTERFACE s_axilite port=request_tag bundle=control offset=0x30
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW
    hls::stream<Segment> segments;
    hls::stream<Result> results;
#pragma HLS STREAM variable=segments depth=4
#pragma HLS STREAM variable=results depth=4
    load_input(input_words, segment_count, segments);
    compute(segments, segment_count, request_tag, results);
    store_output(results, segment_count, output_words);
}
