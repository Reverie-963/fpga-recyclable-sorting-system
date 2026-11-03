#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <stdlib.h>
extern "C" void phase_pixel_accel(const uint32_t*,const uint32_t*,const uint8_t*,const uint32_t*,uint8_t*,uint64_t*,uint32_t*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned&,unsigned&);
static uint32_t seed=8;
static uint32_t random_word(){seed=seed*1664525+1013904223;return seed;}
static void require(bool value,const char *message){if(!value){fprintf(stderr,"FAIL %s\n",message);exit(1);}}
static void check(unsigned w,unsigned h,unsigned masks,unsigned threshold,bool full) {
    // Co-sim adapters snapshot the declared m_axi depth even for a small ROI.
    // Keep every actual allocation as large as its declared interface depth.
    unsigned n=w*h; std::vector<uint32_t>a(307200),b(307200);std::vector<uint8_t>d(307200,99),lut(768);
    uint32_t rois[24],regions[12];uint64_t moments[80];unsigned done=999,status=999;
    for(unsigned c=0;c<3;c++)for(unsigned v=0;v<256;v++)lut[c*256+v]=full?v:(uint8_t)((v*(c+1)+17*c)&255);
    for(unsigned r=0;r<6;r++){rois[r*4]=r%2 ? w/3:0;rois[r*4+1]=0;rois[r*4+2]=w;rois[r*4+3]=h;}
    for(unsigned i=0;i<n;i++){a[i]=full?0:random_word();b[i]=full?0xffffffff:random_word();}
    phase_pixel_accel(a.data(),b.data(),lut.data(),rois,d.data(),moments,regions,w,h,threshold,masks,73,done,status);
    require(done==73 && status==1,"completion");
    std::vector<unsigned> delta(n);uint32_t expected_regions[12]={};
    for(unsigned i=0;i<n;i++){
        unsigned value=0;
        for(unsigned c=0;c<3;c++){int aa=lut[c*256+((a[i]>>(8*c))&255)],bb=(b[i]>>(8*c))&255;value+=(unsigned)abs(aa-bb);}
        delta[i]=value;require(d[i]==(value>threshold),"changed mask strict threshold/full channel sum");
    }
    for(unsigned r=0;r<6;r++){
        for(unsigned y=rois[4*r+1];y<rois[4*r+3];y++)for(unsigned x=rois[4*r];x<rois[4*r+2];x++){
            expected_regions[2*r]+=delta[y*w+x]>threshold;expected_regions[2*r+1]+=delta[y*w+x];
        }
        require(regions[2*r]==expected_regions[2*r] && regions[2*r+1]==expected_regions[2*r+1],"ROI reductions");
    }
    for(unsigned m=0;m<8;m++){
        uint64_t gold[10]={};gold[6]=w;gold[7]=h;
        // Reverse traversal, independent coordinates and arithmetic oracle.
        for(unsigned i=n;i>0;i--){unsigned p=i-1,x=p%w,y=p/w;if(m>=masks || !((b[p]>>(24+m))&1))continue;
            gold[0]++;gold[1]+=x;gold[2]+=y;gold[3]+=(uint64_t)x*x;gold[4]+=(uint64_t)y*y;gold[5]+=(uint64_t)x*y;
            gold[6]=std::min(gold[6],(uint64_t)x);gold[7]=std::min(gold[7],(uint64_t)y);
            gold[8]=std::max(gold[8],(uint64_t)x);gold[9]=std::max(gold[9],(uint64_t)y);
        }
        if(!gold[0])for(unsigned k=6;k<10;k++)gold[k]=0;
        for(unsigned k=0;k<10;k++)require(moments[10*m+k]==gold[k],"exact binary moments inclusive bounds");
        if(full && m<masks)require(gold[3]==(uint64_t)h*(w-1)*w*(2*w-1)/6,"full-frame moment closed form");
    }
}
int main(int argc,char**argv){
    bool small=argc>1;
    check(1,1,0,0,false);check(17,9,8,45,false);check(64,48,1,765,false);
    check(80,60,8,255,false);check(32,24,8,764,true);check(32,24,8,765,true);
    if(!small){check(640,480,8,45,true);check(640,480,8,45,false);}
    std::vector<uint32_t>a(307200),b(307200);std::vector<uint8_t>change(307200,71);
    uint32_t rois[24]={},regions[12];uint8_t lut[768]={};uint64_t moments[80];
    for(unsigned r=0;r<6;r++){rois[4*r+2]=1;rois[4*r+3]=1;}
    for(unsigned mode=0;mode<7;mode++){
        std::fill(moments,moments+80,123);std::fill(regions,regions+12,123);unsigned done=77,status=77;
        if(mode==6)rois[2]=2;
        phase_pixel_accel(a.data(),b.data(),lut,rois,change.data(),moments,regions,mode==0?0:(mode==1?641:1),mode==2?481:1,mode==3?766:45,mode==4?9:8,mode==5?0:73,done,status);
        require(status==4 && done==0,"invalid input cannot publish completion");
        require(change[0]==71 && moments[0]==123 && regions[0]==123,"invalid input leaves payload untouched");
    }
    printf("PASS phase_pixel_accel: %s and seven rejection cases; no hardware acceptance\n",small?"six small frames":"eight frames including 640x480");return 0;
}
