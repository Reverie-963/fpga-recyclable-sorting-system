#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>

extern "C" void station_vision_accel(const uint32_t*,const uint32_t*,uint8_t*,const uint32_t*,uint32_t*,
    unsigned,unsigned,unsigned,unsigned,unsigned,unsigned&,unsigned&);

static uint32_t seed=12345;
static uint32_t random_word(){seed=1664525u*seed+1013904223u;return seed&0xffffffu;}

static bool run(unsigned w,unsigned h,unsigned threshold) {
    // The RTL wrapper snapshots each m_axi port's declared depth, including
    // small-frame transactions. Allocate that full depth in every test case.
    unsigned n=w*h;std::vector<uint32_t> a(307200),b(307200),rect(24),out(48),expected(48);
    std::vector<uint8_t> history(307200,0),gold(307200,0);
    for(unsigned i=0;i<n;i++)a[i]=random_word();
    for(unsigned r=0;r<6;r++){
        rect[r*4]=r%2;rect[r*4+1]=r%2;rect[r*4+2]=w-r%3;rect[r*4+3]=h-r%3;
    }
    for(unsigned frame=0;frame<4;frame++){
        for(unsigned i=0;i<n;i++)b[i]=i%5==frame%5 ? a[i] : random_word();
        std::fill(expected.begin(),expected.end(),0);
        for(unsigned r=0;r<6;r++){
            expected[r*8+3]=w;expected[r*8+4]=h;
            expected[r*8+7]=(rect[r*4+2]-rect[r*4])*(rect[r*4+3]-rect[r*4+1]);
        }
        bool reset=frame==0||frame==3;
        for(unsigned y=0;y<h;y++)for(unsigned x=0;x<w;x++){
            unsigned i=y*w+x,delta=0;
            for(unsigned c=0;c<3;c++){
                int av=(a[i]>>(c*8))&255,bv=(b[i]>>(c*8))&255;
                delta+=av>bv?av-bv:bv-av;
            }
            bool change=delta>threshold;gold[i]=change&&(reset||gold[i]);
            for(unsigned r=0;r<6;r++)if(x>=rect[r*4]&&y>=rect[r*4+1]&&x<rect[r*4+2]&&y<rect[r*4+3]){
                unsigned p=r*8;expected[p]+=change;expected[p+1]+=gold[i];expected[p+2]+=delta;
                if(change){expected[p+3]=std::min(expected[p+3],x);expected[p+4]=std::min(expected[p+4],y);
                    expected[p+5]=std::max(expected[p+5],x);expected[p+6]=std::max(expected[p+6],y);}
            }
        }
        for(unsigned r=0;r<6;r++)if(!expected[r*8])for(unsigned k=3;k<7;k++)expected[r*8+k]=0;
        unsigned job=0,status=0;station_vision_accel(a.data(),b.data(),history.data(),rect.data(),out.data(),
            w,h,threshold,reset,frame+100,job,status);
        if(status!=1||job!=frame+100||out!=expected||history!=gold){
            printf("FAIL %ux%u threshold=%u frame=%u status=%u\n",w,h,threshold,frame,status);return false;}
    }
    return true;
}

int main(int argc,char**argv){
    bool small=argc>1&&!strcmp(argv[1],"--small");
    if(!run(17,11,0)||!run(19,13,255)||!run(9,7,765)||(!small&&!run(640,480,45)))return 1;
    std::vector<uint32_t> a(307200),b(307200),rect(24),out(48);std::vector<uint8_t> history(307200);unsigned job,status;
    station_vision_accel(a.data(),b.data(),history.data(),rect.data(),out.data(),0,1,45,1,7,job,status);
    if(status!=4||job!=7)return 2;
    station_vision_accel(a.data(),b.data(),history.data(),rect.data(),out.data(),1,1,45,1,8,job,status);
    if(status!=4||job!=8)return 3;
    printf("PASS station_vision_accel: RGB sums, six ROIs, window persistence, reset, bounds%s\n",small?" (small RTL cases)":" (including 640x480)");
    return 0;
}
