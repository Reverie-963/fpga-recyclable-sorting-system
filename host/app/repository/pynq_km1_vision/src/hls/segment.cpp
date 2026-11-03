#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include "segment_math.hpp"
typedef ap_axiu<32,0,0,0> axis_word;
void segment(hls::stream<axis_word>& input, hls::stream<axis_word>& output,
             unsigned pixels, unsigned background, unsigned threshold) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE s_axilite port=pixels bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=background bundle=control offset=0x18
#pragma HLS INTERFACE s_axilite port=threshold bundle=control offset=0x20
#pragma HLS INTERFACE s_axilite port=return bundle=control
    for (unsigned i=0;i<pixels;i++) {
#pragma HLS PIPELINE II=1
        axis_word in=input.read(), out;
        out.data=segment_pixel((unsigned)in.data,background,threshold);
        out.keep=15;out.strb=15;out.last=(i==pixels-1);
        output.write(out);
    }
}
