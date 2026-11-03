// File-backed C simulation only; never opens a camera or hardware.
#include <stdint.h>
#include <cstdio>
#include <vector>

extern "C" void station_frame_pipeline(const uint32_t *,const uint32_t *,const uint32_t *,
    uint32_t *,uint32_t *,const uint32_t *,uint32_t *,unsigned,unsigned,unsigned,unsigned,
    unsigned,unsigned,unsigned &,unsigned &);

int main(int argc, char **argv) {
    if(argc!=3) return 1;
    FILE *input=std::fopen(argv[1],"rb"), *output=std::fopen(argv[2],"wb");
    if(!input || !output) return 2;
    uint32_t header[28];
    if(std::fread(header,4,28,input)!=28) return 3;
    unsigned w=header[0],h=header[1],threshold=header[2],frames=header[3];
    if(!w || !h || w>640 || h>480 || !frames || frames>10) return 4;
    unsigned n=w*h,words=(n+31)/32;
    std::vector<uint32_t> base(307200),current(307200),preview(307200),prior(9600),next(9600);
    uint32_t results[88];
    if(std::fread(base.data(),4,n,input)!=n) return 5;
    for(unsigned f=0;f<frames;++f) {
        if(std::fread(current.data(),4,n,input)!=n) return 6;
        unsigned done=0,status=0;
        station_frame_pipeline(base.data(),current.data(),prior.data(),next.data(),preview.data(),
            header+4,results,w,h,threshold,f==0,1,100+f,done,status);
        if(done!=100+f || status!=1) return 7;
        if(std::fwrite(results,4,88,output)!=88 ||
           std::fwrite(next.data(),4,words,output)!=words ||
           std::fwrite(preview.data(),4,n,output)!=n) return 8;
        prior.swap(next);
    }
    std::fclose(input);std::fclose(output);
    std::printf("PASS: recorded scene C simulation, %u frames\n",frames);
    return 0;
}
