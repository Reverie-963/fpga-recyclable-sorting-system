#include <stdint.h>

// Six half-open image rectangles. Input words hold B,G,R in bits 0..23.
// History is the intersection of changed pixels since reset, not a classifier.
// The caller owns baseline/history lifetime and rejects stale frame identities.
extern "C" void station_vision_accel(
    const uint32_t *baseline, const uint32_t *current, uint8_t *history,
    const uint32_t *rois, uint32_t *results,
    unsigned width, unsigned height, unsigned threshold, unsigned reset_history,
    unsigned job_id, unsigned &completed_job, unsigned &status) {
#pragma HLS INTERFACE m_axi port=baseline offset=slave bundle=base_mem depth=307200
#pragma HLS INTERFACE m_axi port=current offset=slave bundle=frame_mem depth=307200
#pragma HLS INTERFACE m_axi port=history offset=slave bundle=history_mem depth=307200
#pragma HLS INTERFACE m_axi port=rois offset=slave bundle=config_mem depth=24
#pragma HLS INTERFACE m_axi port=results offset=slave bundle=config_mem depth=48
#pragma HLS INTERFACE s_axilite port=baseline bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=current bundle=control offset=0x20
#pragma HLS INTERFACE s_axilite port=history bundle=control offset=0x30
#pragma HLS INTERFACE s_axilite port=rois bundle=control offset=0x40
#pragma HLS INTERFACE s_axilite port=results bundle=control offset=0x50
#pragma HLS INTERFACE s_axilite port=width bundle=control offset=0x60
#pragma HLS INTERFACE s_axilite port=height bundle=control offset=0x68
#pragma HLS INTERFACE s_axilite port=threshold bundle=control offset=0x70
#pragma HLS INTERFACE s_axilite port=reset_history bundle=control offset=0x78
#pragma HLS INTERFACE s_axilite port=job_id bundle=control offset=0x80
#pragma HLS INTERFACE s_axilite port=completed_job bundle=control offset=0x88
#pragma HLS INTERFACE s_axilite port=status bundle=control offset=0x90
#pragma HLS INTERFACE s_axilite port=return bundle=control
    completed_job = job_id;
    status = 4;
    if (!width || !height || width > 640 || height > 480 || threshold > 765 || reset_history > 1) return;
    uint32_t rect[6][4];
    uint32_t counters[6][8];
#pragma HLS ARRAY_PARTITION variable=rect complete dim=0
#pragma HLS ARRAY_PARTITION variable=counters complete dim=0
    for (unsigned r=0;r<6;r++) {
        for (unsigned k=0;k<4;k++) rect[r][k]=rois[r*4+k];
        if (rect[r][0]>=rect[r][2] || rect[r][1]>=rect[r][3] || rect[r][2]>width || rect[r][3]>height) return;
        for (unsigned k=0;k<8;k++) counters[r][k]=0;
        counters[r][3]=width; counters[r][4]=height;
        counters[r][7]=(rect[r][2]-rect[r][0])*(rect[r][3]-rect[r][1]);
    }
    unsigned i=0;
    for (unsigned y=0;y<height;y++) {
        for (unsigned x=0;x<width;x++,i++) {
#pragma HLS PIPELINE II=1
            uint32_t a=baseline[i],b=current[i];
            unsigned delta=0;
            for (unsigned channel=0;channel<3;channel++) {
#pragma HLS UNROLL
                unsigned av=(a>>(channel*8))&255,bv=(b>>(channel*8))&255;
                delta += av>bv ? av-bv : bv-av;
            }
            bool change=delta>threshold;
            bool persistent=change && (reset_history || history[i]!=0);
            history[i]=persistent ? 1 : 0;
            for (unsigned r=0;r<6;r++) {
#pragma HLS UNROLL
                if (x>=rect[r][0] && x<rect[r][2] && y>=rect[r][1] && y<rect[r][3]) {
                    counters[r][0]+=change;
                    counters[r][1]+=persistent;
                    counters[r][2]+=delta;
                    if (change) {
                        if (x<counters[r][3]) counters[r][3]=x;
                        if (y<counters[r][4]) counters[r][4]=y;
                        if (x>counters[r][5]) counters[r][5]=x;
                        if (y>counters[r][6]) counters[r][6]=y;
                    }
                }
            }
        }
    }
    for (unsigned r=0;r<6;r++) {
        if (!counters[r][0]) for (unsigned k=3;k<7;k++) counters[r][k]=0;
        for (unsigned k=0;k<8;k++) results[r*8+k]=counters[r][k];
    }
    status=1;
}
