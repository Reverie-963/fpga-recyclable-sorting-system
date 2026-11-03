#include <stdint.h>
#include <ap_int.h>

// Candidate only: BGR bytes 0..23; current bits 24..31 carry up to 8 masks.
// The PS owns the exact uint8 affine LUT, same-frame identity and all geometry
// checks. changed[] remains available to the unchanged CPU 8-connectivity rule.
// This kernel never declares a bin occupied, a material class, or grasp success.
extern "C" void phase_pixel_accel(
    const uint32_t *reference, const uint32_t *current, const uint8_t *lut,
    const uint32_t *rois, uint8_t *changed, uint64_t *moments, uint32_t *regions,
    unsigned width, unsigned height, unsigned threshold, unsigned mask_count,
    unsigned job_id, unsigned &completed_job, unsigned &status) {
#pragma HLS INTERFACE m_axi port=reference offset=slave bundle=frame_a depth=307200
#pragma HLS INTERFACE m_axi port=current offset=slave bundle=frame_b depth=307200
#pragma HLS INTERFACE m_axi port=lut offset=slave bundle=frame_a depth=768
#pragma HLS INTERFACE m_axi port=rois offset=slave bundle=frame_a depth=24
#pragma HLS INTERFACE m_axi port=changed offset=slave bundle=frame_b depth=307200
#pragma HLS INTERFACE m_axi port=moments offset=slave bundle=frame_a depth=80
#pragma HLS INTERFACE m_axi port=regions offset=slave bundle=frame_a depth=12
#pragma HLS INTERFACE s_axilite port=reference bundle=control
#pragma HLS INTERFACE s_axilite port=current bundle=control
#pragma HLS INTERFACE s_axilite port=lut bundle=control
#pragma HLS INTERFACE s_axilite port=rois bundle=control
#pragma HLS INTERFACE s_axilite port=changed bundle=control
#pragma HLS INTERFACE s_axilite port=moments bundle=control
#pragma HLS INTERFACE s_axilite port=regions bundle=control
#pragma HLS INTERFACE s_axilite port=width bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=threshold bundle=control
#pragma HLS INTERFACE s_axilite port=mask_count bundle=control
#pragma HLS INTERFACE s_axilite port=job_id bundle=control
#pragma HLS INTERFACE s_axilite port=completed_job bundle=control
#pragma HLS INTERFACE s_axilite port=status bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    completed_job=0; status=4;
    if (!width || !height || width>640 || height>480 || threshold>765 || mask_count>8 || !job_id) return;
    uint32_t rect[6][4];
#pragma HLS ARRAY_PARTITION variable=rect complete dim=0
    for (unsigned r=0;r<6;r++) {
        for (unsigned k=0;k<4;k++) rect[r][k]=rois[4*r+k];
        if (rect[r][0]>=rect[r][2] || rect[r][1]>=rect[r][3] || rect[r][2]>width || rect[r][3]>height) return;
    }
    uint8_t lookup[3][256];
#pragma HLS ARRAY_PARTITION variable=lookup complete dim=1
    for (unsigned c=0;c<3;c++) {
        for (unsigned v=0;v<256;v++) {
#pragma HLS PIPELINE II=1
            lookup[c][v]=lut[256*c+v];
        }
    }
    // m00,m10,m01,m20,m02,m11,min_col,min_row,max_col,max_row.
    // 640x480 bounds every moment below 2**40; bbox needs only ten bits.
    // The external ABI remains uint64_t, with zero extension at publication.
    ap_uint<40> sums[8][6];
    ap_uint<10> bounds[8][4];
    uint32_t counts[6][2];
#pragma HLS ARRAY_PARTITION variable=sums complete dim=0
#pragma HLS ARRAY_PARTITION variable=bounds complete dim=0
#pragma HLS ARRAY_PARTITION variable=counts complete dim=0
    for (unsigned m=0;m<8;m++) {
        for (unsigned k=0;k<6;k++) sums[m][k]=0;
        bounds[m][0]=width; bounds[m][1]=height;
        bounds[m][2]=0; bounds[m][3]=0;
    }
    for (unsigned r=0;r<6;r++) { counts[r][0]=0; counts[r][1]=0; }
    unsigned index=0;
    for (unsigned y=0;y<height;y++) {
        for (unsigned x=0;x<width;x++,index++) {
#pragma HLS PIPELINE II=1
            uint32_t a=reference[index],b=current[index];
            unsigned delta=0;
            for (unsigned c=0;c<3;c++) {
#pragma HLS UNROLL
                unsigned av=lookup[c][(a>>(8*c))&255],bv=(b>>(8*c))&255;
                delta+=av>bv ? av-bv : bv-av;
            }
            bool different=delta>threshold;
            changed[index]=different ? 1 : 0;
            ap_uint<10> px=x,py=y;
            ap_uint<20> xx=px*px,yy=py*py,xy=px*py;
            for (unsigned m=0;m<8;m++) {
#pragma HLS UNROLL
                if (m<mask_count && ((b>>(24+m))&1)) {
                    sums[m][0]++; sums[m][1]+=x; sums[m][2]+=y;
                    sums[m][3]+=xx; sums[m][4]+=yy; sums[m][5]+=xy;
                    if (px<bounds[m][0]) bounds[m][0]=px;
                    if (py<bounds[m][1]) bounds[m][1]=py;
                    if (px>bounds[m][2]) bounds[m][2]=px;
                    if (py>bounds[m][3]) bounds[m][3]=py;
                }
            }
            for (unsigned r=0;r<6;r++) {
#pragma HLS UNROLL
                if (x>=rect[r][0] && x<rect[r][2] && y>=rect[r][1] && y<rect[r][3]) {
                    counts[r][0]+=different; counts[r][1]+=delta;
                }
            }
        }
    }
    for (unsigned m=0;m<8;m++) {
        if (!sums[m][0]) for (unsigned k=0;k<4;k++) bounds[m][k]=0;
        for (unsigned k=0;k<6;k++) moments[10*m+k]=sums[m][k].to_uint64();
        for (unsigned k=0;k<4;k++) moments[10*m+6+k]=bounds[m][k].to_uint();
    }
    for (unsigned r=0;r<6;r++) for (unsigned k=0;k<2;k++) regions[2*r+k]=counts[r][k];
    completed_job=job_id; status=1;
}
