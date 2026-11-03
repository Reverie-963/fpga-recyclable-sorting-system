#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cassert>
#include <initializer_list>
#include <cstdlib>
typedef ap_axiu<32,0,0,0> axis_word;
void segment(hls::stream<axis_word>&,hls::stream<axis_word>&,unsigned,unsigned,unsigned);
int main(){
    for(unsigned n: {1u,17u,307200u}) {
        hls::stream<axis_word> in,out;
        for(unsigned i=0;i<n;i++){axis_word w;w.data=(i*7193)&0xffffff;w.keep=15;w.strb=15;w.last=i==n-1;in.write(w);}
        segment(in,out,n,0xf0f0f0,100);
        for(unsigned i=0;i<n;i++){
            axis_word w=out.read();unsigned p=(i*7193)&0xffffff;
            int d=std::abs(int(p&255)-240)+std::abs(int((p>>8)&255)-240)+std::abs(int((p>>16)&255)-240);
            assert(w.data==(d>100?255:0));assert(w.keep==15);assert(w.last==(i==n-1));
        }
        assert(out.empty());assert(in.empty());
    }
}
