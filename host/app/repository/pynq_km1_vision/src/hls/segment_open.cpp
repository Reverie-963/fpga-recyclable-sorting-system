#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include "segment_math.hpp"
#include "morph_stream.hpp"
typedef ap_axiu<32,0,0,0> axis_word;
static void threshold_stage(hls::stream<axis_word>& input,hls::stream<unsigned char>& mask,unsigned n,unsigned bg,unsigned threshold){
    for(unsigned i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        mask.write((unsigned char)segment_pixel((unsigned)input.read().data,bg,threshold));
    }
}
static void pack_stage(hls::stream<unsigned char>& mask,hls::stream<axis_word>& output,unsigned n){
    for(unsigned i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        axis_word word;word.data=mask.read();word.keep=15;word.strb=15;word.last=(i==n-1);output.write(word);
    }
}
void segment_open(hls::stream<axis_word>& input,hls::stream<axis_word>& output,
                  unsigned width,unsigned height,unsigned background,unsigned threshold){
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE s_axilite port=width bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=height bundle=control offset=0x18
#pragma HLS INTERFACE s_axilite port=background bundle=control offset=0x20
#pragma HLS INTERFACE s_axilite port=threshold bundle=control offset=0x28
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW
    hls::stream<unsigned char> raw("raw"),eroded("eroded"),opened("opened");
#pragma HLS STREAM variable=raw depth=64
#pragma HLS STREAM variable=eroded depth=64
#pragma HLS STREAM variable=opened depth=64
    threshold_stage(input,raw,width*height,background,threshold);
    morph3<true>(raw,eroded,width,height);
    morph3<false>(eroded,opened,width,height);
    pack_stage(opened,output,width*height);
}
