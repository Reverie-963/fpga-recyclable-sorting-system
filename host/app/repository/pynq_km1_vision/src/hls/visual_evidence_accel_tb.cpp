// visual_evidence_accel testbench：CPU 参考实现逐项对比
// 编译: g++ -std=c++17 visual_evidence_accel.cpp visual_evidence_accel_tb.cpp -I <Vitis>/include
// 运行: ./a.exe（需 tps\mingw bin 在 PATH 以加载运行库）

#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef ap_axiu<32,0,0,0> axis_word;

// 核函数原型（核在 visual_evidence_accel.cpp，无独立头文件）
extern void visual_evidence_accel(
    hls::stream<axis_word>& inputA, hls::stream<axis_word>& inputB, hls::stream<axis_word>& output,
    unsigned width, unsigned height, unsigned threshold, unsigned mode, unsigned job_id,
    ap_uint<32>& stat0_lo, ap_uint<32>& stat0_hi,
    ap_uint<32>& stat1_lo, ap_uint<32>& stat1_hi,
    ap_uint<32>& stat2_lo, ap_uint<32>& stat2_hi,
    ap_uint<32>& stat3_lo, ap_uint<32>& stat3_hi,
    unsigned& min_row, unsigned& max_row, unsigned& min_col, unsigned& max_col,
    unsigned& status);

// CPU 参考：统计 [s0..s3]、包围框、change 掩码
static void ref_evidence(const std::vector<unsigned char>& A,
                         const std::vector<unsigned char>& B,
                         unsigned width, unsigned height,
                         unsigned threshold, unsigned mode,
                         ap_uint<64> s[4], unsigned bbox[4],
                         std::vector<unsigned char>& change_out){
    const unsigned n = width*height;
    ap_uint<64> s0=0,s1=0,s2=0,s3=0;
    unsigned rmin=height, rmax=0, cmin=width, cmax=0;
    bool any=false;
    bool mask_mode = (mode!=0);
    change_out.assign(n,0);
    for(unsigned i=0;i<n;i++){
        unsigned y=i/width, x=i%width;
        unsigned char a=A[i], b=B[i];
        bool change=false;
        if(!mask_mode){
            unsigned char diff = (a>b)?(a-b):(b-a);
            change = (diff > (unsigned char)threshold);
            if(change) s0 += 1;
        } else {
            unsigned av=(a!=0)?1:0, bv=(b!=0)?1:0;
            s3 += (av|bv);
            s2 += (bv & ~av);
            s1 += (av & ~bv);
            s0 += (av & bv);
            change = (av!=bv);
        }
        change_out[i] = change ? 1 : 0;
        if(change){
            any=true;
            if(x<cmin) cmin=x; if(x>cmax) cmax=x;
            if(y<rmin) rmin=y; if(y>rmax) rmax=y;
        }
    }
    if(!any){ rmin=0; rmax=0; cmin=0; cmax=0; }
    s[0]=s0; s[1]=s1; s[2]=s2; s[3]=s3;
    bbox[0]=rmin; bbox[1]=rmax; bbox[2]=cmin; bbox[3]=cmax;
}

static int run_case(const char* name,
                    std::vector<unsigned char>& A,
                    std::vector<unsigned char>& B,
                    unsigned width, unsigned height,
                    unsigned threshold, unsigned mode,
                    unsigned expect_status){
    hls::stream<axis_word> ia("ia"), ib("ib"), o("o");
    for(size_t i=0; i<A.size() && !(expect_status & 4); i++){
        axis_word wa,wb; wa.data=A[i]; wb.data=B[i]; wa.keep=wb.keep=15; wa.strb=wb.strb=15;
        wa.last=wb.last=(i==A.size()-1); ia.write(wa); ib.write(wb);
    }
    unsigned w0=width,h0=height,th0=threshold,md0=mode,jb0=0x1234;
    ap_uint<32> s0l,s0h,s1l,s1h,s2l,s2h,s3l,s3h;
    unsigned mr,mx,mc,Mc, st=0;
    visual_evidence_accel(ia,ib,o,w0,h0,th0,md0,jb0,
                          s0l,s0h,s1l,s1h,s2l,s2h,s3l,s3h,
                          mr,mx,mc,Mc,st);

    // 期望格式错误
    if(expect_status & 0x4u){
        if((st & 0x4u)==0){ printf("FAIL %-24s 期望FORMAT_ERROR未置位\n",name); return 1; }
        printf("PASS %-24s (FORMAT_ERROR)\n", name);
        return 0;
    }

    // 消费并校验输出流 change 掩码
    std::vector<unsigned char> change_got;
    bool framing_ok=true;
    for(size_t i=0;i<A.size();i++){
        axis_word wo = o.read(); change_got.push_back((unsigned char)wo.data);
        if(wo.keep!=15 || wo.strb!=15 || wo.last!=(i==A.size()-1)) framing_ok=false;
    }

    // CPU 参考
    ap_uint<64> ref[4]; unsigned rbb[4]; std::vector<unsigned char> ref_change;
    ref_evidence(A,B,width,height,threshold,mode,ref,rbb,ref_change);

    ap_uint<64> got[4];
    got[0]=((ap_uint<64>)s0h<<32)|s0l;
    got[1]=((ap_uint<64>)s1h<<32)|s1l;
    got[2]=((ap_uint<64>)s2h<<32)|s2l;
    got[3]=((ap_uint<64>)s3h<<32)|s3l;
    bool ok=framing_ok;
    for(int k=0;k<4;k++) if(got[k]!=ref[k]) ok=false;
    unsigned gbb[4]={mr,mx,mc,Mc};
    for(int k=0;k<4;k++) if(gbb[k]!=rbb[k]) ok=false;
    if(change_got != ref_change) ok=false;
    if(st != 0x1u) ok=false;

    if(ok){
        printf("PASS %-24s s=[%llu %llu %llu %llu] bbox=[%u %u %u %u]\n",
               name,(unsigned long long)got[0],(unsigned long long)got[1],
               (unsigned long long)got[2],(unsigned long long)got[3],
               gbb[0],gbb[1],gbb[2],gbb[3]);
        return 0;
    }
    printf("FAIL %-24s\n  核:  s=[%llu %llu %llu %llu] bbox=[%u %u %u %u]\n  参考:s=[%llu %llu %llu %llu] bbox=[%u %u %u %u]\n",
           name,(unsigned long long)got[0],(unsigned long long)got[1],
           (unsigned long long)got[2],(unsigned long long)got[3],
           gbb[0],gbb[1],gbb[2],gbb[3],
           (unsigned long long)ref[0],(unsigned long long)ref[1],
           (unsigned long long)ref[2],(unsigned long long)ref[3],
           rbb[0],rbb[1],rbb[2],rbb[3]);
    return 1;
}

int main(){
    int fails=0;
    unsigned w,h;
    std::vector<unsigned char> A,B;

    // ---- 灰度模式 ----
    w=4; h=3; A={10,20,30,40, 50,60,70,80, 90,100,110,120}; B=A;
    fails+=run_case("gray-identical",A,B,w,h,20,0,0);
    w=4; h=3; A={10,20,30,40, 50,60,70,80, 90,100,110,120};
    B={10,20,30,40, 50,99,70,80, 90,100,110,120};
    fails+=run_case("gray-single-diff",A,B,w,h,20,0,0);
    w=5; h=4; A.resize(20); B.resize(20);
    for(int i=0;i<20;i++){ A[i]=i*5; B[i]=(i%7==0)?(A[i]+40):A[i]; }
    fails+=run_case("gray-multi-diff",A,B,w,h,30,0,0);
    w=200; h=150; A.resize(30000); B.resize(30000);
    srand(7); for(int i=0;i<30000;i++){ A[i]=rand()%256; B[i]=rand()%256; }
    fails+=run_case("gray-large-random",A,B,w,h,64,0,0);
    w=3; h=2; A={1,2,3, 4,5,6}; B={1,9,3, 4,5,6};
    fails+=run_case("gray-thresh0",A,B,w,h,0,0,0);
    w=3; h=2; A={1,2,3, 4,5,6}; B={1,9,3, 4,5,6};
    fails+=run_case("gray-thresh255",A,B,w,h,255,0,0);

    // ---- 掩码模式 ----
    w=4; h=3; A={0,0,0,0, 0,0,0,0, 0,0,0,0}; B=A;
    fails+=run_case("mask-allzero",A,B,w,h,0,1,0);
    w=4; h=3; A={1,1,1,1, 1,1,1,1, 1,1,1,1}; B=A;
    fails+=run_case("mask-allone-same",A,B,w,h,0,1,0);
    w=4; h=3; A={1,1,1,1, 1,1,1,1, 1,1,1,1}; B={0,0,0,0, 0,0,0,0, 0,0,0,0};
    fails+=run_case("mask-Aall-Bnone",A,B,w,h,0,1,0);
    w=4; h=3; A={0,0,0,0, 0,0,0,0, 0,0,0,0}; B={1,1,1,1, 1,1,1,1, 1,1,1,1};
    fails+=run_case("mask-Anone-Ball",A,B,w,h,0,1,0);
    w=8; h=6; A.resize(48); B.resize(48);
    srand(11); for(int i=0;i<48;i++){ A[i]=rand()%2; B[i]=rand()%2; }
    fails+=run_case("mask-random-overlap",A,B,w,h,0,1,0);
    w=4; h=3; A={0,0,0,0, 0,0,0,0, 0,0,0,0};
    B={0,0,0,0, 0,0,0,0, 0,0,0,1};
    fails+=run_case("mask-corner-change",A,B,w,h,0,1,0);
    w=3; h=2; A={5,0,9, 0,2,0}; B={0,0,9, 0,0,0};
    fails+=run_case("mask-nonbinary",A,B,w,h,0,1,0);
    // 非方形回归：宽扁触发 cmin 初始化（cmin 曾=height 错误），窄高触发 rmin（曾=width 错误）
    w=5; h=2; A={0,0,0,0,0, 0,0,0,0,0}; B={0,0,0,0,1, 0,0,0,0,0};
    fails+=run_case("mask-nonsquare-wide-cmin",A,B,w,h,0,1,0);
    w=2; h=5; A={0,0, 0,0, 0,0, 0,0, 0,0}; B={0,0, 0,0, 0,0, 0,0, 0,1};
    fails+=run_case("mask-nonsquare-tall-rmin",A,B,w,h,0,1,0);
    w=300; h=200; A.resize(60000); B.resize(60000);
    srand(14); for(int i=0;i<60000;i++){ A[i]=rand()%2; B[i]=rand()%2; }
    fails+=run_case("mask-large-random",A,B,w,h,0,1,0);

    // ---- 格式错误 ----
    w=640; h=480; A.assign(w*h,0); B.assign(w*h,255);
    fails+=run_case("max-frame-gray",A,B,w,h,254,0,0);
    fails+=run_case("max-frame-mask",A,B,w,h,0,1,0);
    w=1; h=480; A.assign(h,255); B.assign(h,0); B[h-1]=255;
    fails+=run_case("single-column-mask",A,B,w,h,0,1,0);
    w=0; h=3; A={0,0,0,0,0,0,0,0,0,0,0,0}; B=A;
    fails+=run_case("format-zero-width",A,B,w,h,0,0,0x4u);
    w=700; h=3; A.assign(2100,0); B=A;
    fails+=run_case("format-over-width",A,B,w,h,0,0,0x4u);
    w=3; h=0; A.assign(12,0); B=A;
    fails+=run_case("format-zero-height",A,B,w,h,0,0,0x4u);

    printf("=== 全部%s通过 === failures=%d\n", fails==0?"":"未", fails);
    return fails==0 ? 0 : 1;
}
