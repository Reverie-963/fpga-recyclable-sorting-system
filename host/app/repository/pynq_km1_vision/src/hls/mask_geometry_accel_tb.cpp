#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>
#include <vector>
#include <cstdlib>

typedef ap_axiu<32,0,0,0> axis_word;

// 顶层原型（定义在 mask_geometry_accel.cpp）
void mask_geometry_accel(
    hls::stream<axis_word>& input, hls::stream<axis_word>& output,
    unsigned width, unsigned height,
    unsigned erode_enable, unsigned dilate_enable, unsigned job_id,
    ap_uint<32>& m00_lo, ap_uint<32>& m00_hi,
    ap_uint<32>& m10_lo, ap_uint<32>& m10_hi,
    ap_uint<32>& m01_lo, ap_uint<32>& m01_hi,
    ap_uint<32>& m20_lo, ap_uint<32>& m20_hi,
    ap_uint<32>& m02_lo, ap_uint<32>& m02_hi,
    ap_uint<32>& m11_lo, ap_uint<32>& m11_hi,
    unsigned& min_row, unsigned& max_row, unsigned& min_col, unsigned& max_col,
    unsigned& status);

struct RefResult {
    unsigned long long m00,m10,m01,m20,m02,m11;
    int rmin,rmax,cmin,cmax;
};

// CPU 参考：二值化(已是0/1) + 可选开运算 + 二阶矩/边界
static RefResult reference(const std::vector<unsigned char>& in_mask, int W, int H,
                           bool erode_en, bool dilate_en, std::vector<unsigned char>& out_mask){
    std::vector<unsigned char> ero(W*H,0);
    if(erode_en){
        for(int y=0;y<H;y++)for(int x=0;x<W;x++){
            bool all=true;
            for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
                int yy=y+dy, xx=x+dx;
                unsigned char val = (yy>=0&&yy<H&&xx>=0&&xx<W)? in_mask[yy*W+xx] : 0;
                if(!val){ all=false; break; }
            }
            ero[y*W+x] = all?1:0;
        }
    } else { ero = in_mask; for(auto& v:ero) v=(v!=0); }

    std::vector<unsigned char> opn(W*H,0);
    if(dilate_en){
        for(int y=0;y<H;y++)for(int x=0;x<W;x++){
            bool any=false;
            for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
                int yy=y+dy, xx=x+dx;
                unsigned char val = (yy>=0&&yy<H&&xx>=0&&xx<W)? ero[yy*W+xx] : 0;
                if(val){ any=true; break; }
            }
            opn[y*W+x] = any?1:0;
        }
    } else opn = ero;

    RefResult r; r.m00=r.m10=r.m01=r.m20=r.m02=r.m11=0;
    r.rmin=H; r.rmax=-1; r.cmin=W; r.cmax=-1;
    out_mask = opn;
    for(int y=0;y<H;y++)for(int x=0;x<W;x++){
        if(opn[y*W+x]){
            unsigned long long ux=x, uy=y;
            r.m00++; r.m10+=ux; r.m01+=uy; r.m20+=ux*ux; r.m02+=uy*uy; r.m11+=ux*uy;
            if(x<r.cmin) r.cmin=x; if(x>r.cmax) r.cmax=x;
            if(y<r.rmin) r.rmin=y; if(y>r.rmax) r.rmax=y;
        }
    }
    if(r.rmin==H){ r.rmin=r.rmax=r.cmin=r.cmax=0; }  // 空掩码约定
    return r;
}

static int failures = 0;

static void run_case(const std::vector<unsigned char>& img, int W, int H,
                     unsigned erode_en, unsigned dilate_en, const char* name){
    hls::stream<axis_word> input("in"), output("out");
    for(int i=0;i<W*H;i++){
        axis_word w; w.data = img[i]; w.keep=15; w.strb=15; w.last=(i==W*H-1)?1:0;
        input.write(w);
    }
    ap_uint<32> m00_lo,m00_hi,m10_lo,m10_hi,m01_lo,m01_hi,m20_lo,m20_hi,m02_lo,m02_hi,m11_lo,m11_hi;
    unsigned minr,maxr,minc,maxc,status;
    mask_geometry_accel(input, output, W, H, erode_en, dilate_en, 7,
        m00_lo,m00_hi, m10_lo,m10_hi, m01_lo,m01_hi,
        m20_lo,m20_hi, m02_lo,m02_hi, m11_lo,m11_hi,
        minr,maxr,minc,maxc,status);

    std::vector<unsigned char> got(W*H,0);
    bool framing_ok=true;
    for(int i=0;i<W*H;i++){
        axis_word w = output.read(); got[i] = (unsigned char)w.data;
        if(w.keep!=15 || w.strb!=15 || w.last!=(i==W*H-1)) framing_ok=false;
    }

    std::vector<unsigned char> exp;
    RefResult r = reference(img, W, H, erode_en!=0, dilate_en!=0, exp);

    unsigned long long hm00 = ((unsigned long long)(unsigned)m00_hi<<32)|(unsigned)m00_lo;
    unsigned long long hm10 = ((unsigned long long)(unsigned)m10_hi<<32)|(unsigned)m10_lo;
    unsigned long long hm01 = ((unsigned long long)(unsigned)m01_hi<<32)|(unsigned)m01_lo;
    unsigned long long hm20 = ((unsigned long long)(unsigned)m20_hi<<32)|(unsigned)m20_lo;
    unsigned long long hm02 = ((unsigned long long)(unsigned)m02_hi<<32)|(unsigned)m02_lo;
    unsigned long long hm11 = ((unsigned long long)(unsigned)m11_hi<<32)|(unsigned)m11_lo;

    bool ok = framing_ok && (got==exp)
           && (hm00==r.m00) && (hm10==r.m10) && (hm01==r.m01)
           && (hm20==r.m20) && (hm02==r.m02) && (hm11==r.m11)
           && (minr==(unsigned)r.rmin) && (maxr==(unsigned)r.rmax)
           && (minc==(unsigned)r.cmin) && (maxc==(unsigned)r.cmax)
           && (status == 0x1);

    if(!ok){
        failures++;
        std::cout << "FAIL: " << name << std::endl;
        std::cout << "  m00 hw=" << hm00 << " ref=" << r.m00
                  << "  minr/maxr hw=" << minr << "/" << maxr << " ref=" << r.rmin << "/" << r.rmax
                  << "  minc/maxc hw=" << minc << "/" << maxc << " ref=" << r.cmin << "/" << r.cmax << std::endl;
    } else {
        std::cout << "PASS: " << name << std::endl;
    }
}

int main(){
    std::cout << "=== mask_geometry_accel C 仿真 ===" << std::endl;

    // 1. 全零掩码（空）
    run_case(std::vector<unsigned char>(20*15,0), 20, 15, 1, 1, "empty-all-zero");

    // 2. 全一掩码（开运算会收缩）
    run_case(std::vector<unsigned char>(20*15,1), 20, 15, 1, 1, "all-one-open");

    // 3. 全一掩码，开运算旁路
    run_case(std::vector<unsigned char>(20*15,1), 20, 15, 0, 0, "all-one-passthrough");

    // 4. 固定随机掩码（种子 42）
    {
        std::vector<unsigned char> img(20*15);
        std::srand(42);
        for(size_t i=0;i<img.size();i++) img[i] = (std::rand()%100<40)?1:0;
        run_case(img, 20, 15, 1, 1, "random-open");
        run_case(img, 20, 15, 0, 0, "random-passthrough");
        run_case(img, 20, 15, 1, 0, "random-erode-only");
        run_case(img, 20, 15, 0, 1, "random-dilate-only");
    }

    // 5. 单像素
    {
        std::vector<unsigned char> img(20*15,0); img[7*20+8]=1;
        run_case(img, 20, 15, 1, 1, "single-pixel-open");  // 开运算会抹掉
        run_case(img, 20, 15, 0, 1, "single-pixel-dilate");
        run_case(img, 20, 15, 0, 0, "single-pixel-passthrough");
    }

    // 6. 3x3 实心块（开运算应保留）
    {
        std::vector<unsigned char> img(20*15,0);
        for(int y=5;y<=7;y++)for(int x=5;x<=7;x++) img[y*20+x]=1;
        run_case(img, 20, 15, 1, 1, "3x3-block-open");
    }

    // 7. 十字图案
    {
        std::vector<unsigned char> img(20*15,0);
        int cx=10, cy=7;
        for(int x=cx-2;x<=cx+2;x++) img[cy*20+x]=1;
        for(int y=cy-2;y<=cy+2;y++) img[y*20+cx]=1;
        run_case(img, 20, 15, 1, 1, "cross-open");
    }

    // 8. 边界角像素（腐蚀/膨胀的补零边界）
    {
        std::vector<unsigned char> img(20*15,0);
        img[0]=1; // 左上角
        img[14*20+19]=1; // 右下角
        run_case(img, 20, 15, 1, 0, "corner-erode");
        run_case(img, 20, 15, 0, 1, "corner-dilate");
        run_case(img, 20, 15, 1, 1, "corner-open");
    }

    // 9. 较大随机（接近 640x480 的工作，用 200x150 验证规模）
    {
        int W=200, H=150;
        std::vector<unsigned char> img(W*H);
        std::srand(123);
        for(size_t i=0;i<img.size();i++) img[i] = (std::rand()%100<35)?1:0;
        run_case(img, W, H, 1, 1, "large-random-open");
        run_case(img, W, H, 0, 0, "large-random-passthrough");
    }

    // 10. 非方形回归：前景集中在宽扁右侧（cmin 曾=height 错误）与窄高底部（rmin 曾=width 错误）
    {
        std::vector<unsigned char> img(20*15,0);
        for(int y=3;y<=12;y++)for(int x=16;x<=19;x++) img[y*20+x]=1; // 列16-19 > height15 → 触发 cmin
        run_case(img,20,15,0,0,"nonsquare-wide-right");
        run_case(img,20,15,1,1,"nonsquare-wide-right-open");
    }
    {
        std::vector<unsigned char> img(5*20,0);
        for(int y=16;y<=18;y++)for(int x=2;x<=3;x++) img[y*5+x]=1; // 行16-18 > width5 → 触发 rmin
        run_case(img,5,20,0,0,"nonsquare-tall-bottom");
    }

    // 11. 格式错误：非法尺寸
    for(unsigned flags=0; flags<4; ++flags){
        run_case(std::vector<unsigned char>(65,255),65,1,flags&1,flags>>1,"fifo-boundary-65");
        run_case(std::vector<unsigned char>(480,255),1,480,flags&1,flags>>1,"single-column");
        std::vector<unsigned char> full(640*480,255);
        for(unsigned y=0;y<480;y++) full[y*640+(y%640)]=0;
        run_case(full,640,480,flags&1,flags>>1,"max-frame-and-last-row");
    }
    {
        hls::stream<axis_word> input("in"), output("out");
        ap_uint<32> m00_lo,m00_hi,m10_lo,m10_hi,m01_lo,m01_hi,m20_lo,m20_hi,m02_lo,m02_hi,m11_lo,m11_hi;
        unsigned minr,maxr,minc,maxc,status;
        mask_geometry_accel(input, output, 0, 100, 1, 1, 7,
            m00_lo,m00_hi,m10_lo,m10_hi,m01_lo,m01_hi,m20_lo,m20_hi,m02_lo,m02_hi,m11_lo,m11_hi,
            minr,maxr,minc,maxc,status);
        if(status & 0x4) std::cout << "PASS: format-error-zero-width" << std::endl;
        else { failures++; std::cout << "FAIL: format-error-zero-width" << std::endl; }
    }

    std::cout << (failures? "=== 有失败 ===" : "=== 全部通过 ===") << " failures=" << failures << std::endl;
    return failures ? 1 : 0;
}
