#include <stdint.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>

extern "C" void station_frame_pipeline(const uint32_t *,const uint32_t *,const uint32_t *,
    uint32_t *,uint32_t *,const uint32_t *,uint32_t *,unsigned,unsigned,unsigned,unsigned,
    unsigned,unsigned,unsigned &,unsigned &);

static const unsigned MAX_N=640*480;
static uint32_t base[MAX_N],frame[MAX_N],preview[MAX_N],prior[9600],next_hist[9600],results[88],rois[24];
static uint32_t seed=1234567;
static uint32_t random_word(){seed=seed*1664525+1013904223;return seed;}
static int sample(const std::vector<unsigned> &a,int x,int y,unsigned w,unsigned h) {
    return x<0 || y<0 || x>=int(w) || y>=int(h) ? 0 : a[y*w+x];
}

static bool run(unsigned w,unsigned h,unsigned round) {
    unsigned n=w*h,words=(n+31)/32,threshold=round==2?765:45;
    unsigned reset=(round==0 || round==3),enabled=round!=2;
    for(unsigned i=0;i<n;++i) {
        if(round==0) base[i]=0;
        frame[i]=round==0 ? 0 : round==1 ? random_word()&0xffffff : round==2 ? base[i] : 0;
    }
    if(round==3) {
        // Drain and row-boundary cases: changes at every image edge and corner.
        for(unsigned i=0;i<n;++i) if(i<w || i>=n-w || i%w==0 || i%w==w-1) frame[i]=0xffffff;
    }
    for(unsigned r=0;r<6;++r) {
        rois[4*r]=(r&1)?w/2:0;rois[4*r+1]=(r&2)?h/2:0;
        rois[4*r+2]=w;rois[4*r+3]=h;
    }
    std::vector<unsigned> gray(n),smooth(n),edge(n),changed(n),persistent(n),eroded(n),clean(n);
    for(unsigned i=0;i<n;++i) {
        unsigned b=frame[i]&255,g=(frame[i]>>8)&255,r=(frame[i]>>16)&255,delta=0;
        gray[i]=(29*b+150*g+77*r+128)>>8;
        for(unsigned k=0;k<3;++k) delta+=std::abs(int((frame[i]>>(8*k))&255)-int((base[i]>>(8*k))&255));
        changed[i]=delta>threshold;
        persistent[i]=changed[i] && (reset || ((prior[i/32]>>(i%32))&1));
    }
    for(unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x) {
        unsigned sum=0,all=1;
        for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
            sum+=sample(gray,int(x)+dx,int(y)+dy,w,h);
            all &= sample(changed,int(x)+dx,int(y)+dy,w,h);
        }
        smooth[y*w+x]=(sum+4)/9;eroded[y*w+x]=all;
    }
    for(unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x) {
        int gx=0,gy=0;unsigned any=0;
        for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
            int v=sample(smooth,int(x)+dx,int(y)+dy,w,h);
            gx+=v*dx*(dy==0?2:1);gy+=v*dy*(dx==0?2:1);
            any |= sample(eroded,int(x)+dx,int(y)+dy,w,h);
        }
        edge[y*w+x]=std::min(255,std::abs(gx)+std::abs(gy));clean[y*w+x]=any;
    }
    uint32_t expected[88]={},expected_history[9600]={};
    for(unsigned r=0;r<6;++r) {expected[r*8+3]=w;expected[r*8+4]=h;expected[r*8+7]=(w-rois[r*4])*(h-rois[r*4+1]);}
    for(unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x) {
        unsigned i=y*w+x,delta=0;
        for(unsigned k=0;k<3;++k) delta+=std::abs(int((frame[i]>>(8*k))&255)-int((base[i]>>(8*k))&255));
        expected_history[i/32] |= persistent[i]<<(i%32);
        ++expected[48+(gray[i]>>4)];
        for(unsigned r=0;r<6;++r) if(x>=rois[r*4] && y>=rois[r*4+1]) {
            unsigned j=r*8;
            expected[j]+=changed[i];expected[j+1]+=persistent[i];expected[j+2]+=delta;
            if(changed[i]) {expected[j+3]=std::min(expected[j+3],x);expected[j+4]=std::min(expected[j+4],y);
                expected[j+5]=std::max(expected[j+5],x);expected[j+6]=std::max(expected[j+6],y);}
            expected[64+r*4]+=gray[i];expected[65+r*4]+=edge[i]>=64;
            expected[66+r*4]+=clean[i];expected[67+r*4]+=smooth[i];
        }
    }
    for(unsigned r=0;r<6;++r) if(!expected[r*8]) for(unsigned k=3;k<7;++k) expected[r*8+k]=0;
    std::fill(preview,preview+MAX_N,0xdeadbeef);std::fill(next_hist,next_hist+9600,0xdeadbeef);
    unsigned completed=0,status=0,job=100+round;
    station_frame_pipeline(base,frame,prior,next_hist,preview,rois,results,w,h,threshold,reset,enabled,job,completed,status);
    if(completed!=job || status!=1) {std::printf("FAIL job/status %u %u\n",completed,status);return false;}
    for(unsigned k=0;k<88;++k) if(results[k]!=expected[k]) {
        std::printf("FAIL result %ux%u round %u word %u: %u != %u\n",w,h,round,k,results[k],expected[k]);return false;
    }
    for(unsigned i=0;i<words;++i) if(next_hist[i]!=expected_history[i]) {
        std::printf("FAIL history %ux%u word %u\n",w,h,i);return false;
    }
    for(unsigned i=0;i<n;++i) {
        uint32_t value=enabled ? gray[i]|(smooth[i]<<8)|(edge[i]<<16)|((clean[i]?255U:0U)<<24) : 0xdeadbeef;
        if(preview[i]!=value) {
            std::printf("FAIL preview %ux%u round %u pixel %u: %08x != %08x\n",w,h,round,i,preview[i],value);return false;
        }
    }
    if(n<MAX_N && preview[n]!=0xdeadbeef) return false;
    if(words<9600 && next_hist[words]!=0xdeadbeef) return false;
    std::copy(next_hist,next_hist+words,prior);
    return true;
}

int main(int argc,char **argv) {
    bool small=argc>1 && std::strcmp(argv[1],"--small")==0;
    const unsigned sizes[][2]={{1,1},{7,5},{17,11},{1,7},{7,1},{2,2},{31,7},{35,19},{640,480}};
    unsigned cases=small?3:9;
    for(unsigned c=0;c<cases;++c) {
        std::fill(prior,prior+9600,0);
        for(unsigned round=0;round<4;++round) if(!run(sizes[c][0],sizes[c][1],round)) return 1;
    }
    std::fill(results,results+88,0xdeadbeef);
    unsigned done=5,status=0;
    station_frame_pipeline(base,frame,prior,next_hist,preview,rois,results,0,1,45,1,1,99,done,status);
    if(done!=0 || status!=4 || results[0]!=0xdeadbeef) return 2;
    std::printf("PASS: %u frame transactions, four preview planes, 88 words, packed history, invalid input\n",cases*4);
    return 0;
}
