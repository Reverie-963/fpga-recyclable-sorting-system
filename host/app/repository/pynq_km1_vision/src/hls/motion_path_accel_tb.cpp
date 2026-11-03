#include "motion_path_accel.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

static double reference(const uint32_t *p, int sample) {
    double angle[3];
    for (int j = 0; j < 3; ++j) {
        double pwm = double(p[j+1])+(double(p[j+7])-p[j+1])*sample/100;
        angle[j] = (pwm-1500)*270/2000;
    }
    double a = (90+angle[0])*std::acos(-1.)/180;
    double b = a-angle[1]*std::acos(-1.)/180;
    double c = b+angle[2]*std::acos(-1.)/180;
    return 100+105*std::sin(a)+88*std::sin(b)+155*std::sin(c);
}

int main(int argc, char **argv) {
    bool small = argc > 1 && std::strcmp(argv[1], "--small") == 0;
    uint32_t input[768], output[512];
    unsigned seed = 20261030, tested = 0;
    double maximum_width = 0;
    unsigned rounds = small ? 2 : 64;
    for (unsigned round = 0; round < rounds; ++round) {
        unsigned n = small ? 2 : MOTION_MAX_SEGMENTS;
        for (unsigned i = 0; i < n*12; ++i) {
            seed = 1664525*seed+1013904223;
            input[i] = 500+seed%2001;
        }
        // Endpoint, no-motion, wrap, maximum-delta and signed-height cases.
        if (round == 0) {
            for (unsigned i = 0; i < 12; ++i) input[i] = 1500;
            for (unsigned i = 12; i < 24; ++i) input[i] = i < 18 ? 500 : 2500;
        }
        uint32_t tag = 0xa1000000+round;
        motion_path_accel(input, output, n, tag);
        for (unsigned row = 0; row < n; ++row) {
            double expected = 1e10;
            unsigned cost = 0;
            for (int i = 0; i <= 100; ++i) {
                double z = reference(input+row*12, i);
                if (z < expected) expected = z;
            }
            for (unsigned j = 0; j < 6; ++j) {
                int d = int(input[row*12+6+j])-int(input[row*12+j]);
                cost += d < 0 ? -d : d;
            }
            const uint32_t *r = output+row*8;
            double lower = int32_t(r[2])/65536., upper = int32_t(r[3])/65536.;
            if (r[0] != MOTION_MAGIC || r[1] != 0 || r[4] > 100 ||
                r[5] != 101 || r[6] != cost || r[7] != tag ||
                lower > expected || upper < expected || upper-lower > .004) {
                std::printf("FAIL row=%u expected=%.10f lower=%.10f upper=%.10f\n",
                            row, expected, lower, upper);
                return 1;
            }
            if (upper-lower > maximum_width) maximum_width = upper-lower;
            ++tested;
        }
    }
    for (unsigned bad = 0; bad < 12; ++bad) {
        for (unsigned j = 0; j < 12; ++j) input[j] = 1500;
        input[bad] = bad & 1 ? 0xffffffff : 499;
        motion_path_accel(input, output, 1, 0xbad);
        if (output[1] != 1 || output[5] != 0 || output[2] != 0x80000000 || output[7] != 0xbad) return 2;
    }
    const unsigned invalid_counts[] = {0, 65, 0xffffffff};
    for (unsigned n : invalid_counts) {
        for (unsigned &word : output) word = 0xdeadbeef;
        motion_path_accel(input, output, n, 42);
        for (unsigned word : output) if (word != 0xdeadbeef) return 3;
    }
    std::printf("PASS %u segments, %u sampled poses, outward width <= %.9f mm; invalid PWM/count pass\n",
                tested, tested*101, maximum_width);
    return 0;
}
