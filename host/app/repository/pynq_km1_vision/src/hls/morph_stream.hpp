#ifndef MORPH_STREAM_HPP
#define MORPH_STREAM_HPP
// Stream must provide read()/write(). Same algorithm runs with HLS or portable queues.
// Zero-valued padding for BOTH erosion and dilation; max width 1920.
template<bool ERODE,typename Stream>
void morph3(Stream& input,Stream& output,unsigned width,unsigned height){
#pragma HLS INLINE off
    unsigned char lines[2][1921];
    unsigned char window[3][3];
#pragma HLS ARRAY_PARTITION variable=lines complete dim=1
#pragma HLS ARRAY_PARTITION variable=window complete dim=0
    for(unsigned x=0;x<=width;x++){
#pragma HLS PIPELINE II=1
        lines[0][x]=0;lines[1][x]=0;
    }
    for(unsigned y=0;y<=height;y++){
        for(unsigned row=0;row<3;row++)
            for(unsigned col=0;col<3;col++)window[row][col]=0;
        for(unsigned x=0;x<=width;x++){
#pragma HLS PIPELINE II=1
            unsigned char bottom=(y<height && x<width)?input.read():0;
            unsigned char top=lines[0][x],middle=lines[1][x];
            lines[0][x]=middle;lines[1][x]=bottom;
            for(unsigned row=0;row<3;row++){
#pragma HLS UNROLL
                window[row][0]=window[row][1];window[row][1]=window[row][2];
            }
            window[0][2]=top;window[1][2]=middle;window[2][2]=bottom;
            if(y>0 && x>0){
                unsigned char result=ERODE?255:0;
                for(unsigned row=0;row<3;row++){
#pragma HLS UNROLL
                    for(unsigned col=0;col<3;col++){
#pragma HLS UNROLL
                        if(ERODE)result=result & window[row][col];
                        else result=result | window[row][col];
                    }
                }
                output.write(result);
            }
        }
    }
}
#endif
