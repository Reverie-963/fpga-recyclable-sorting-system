#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include "morph_stream.hpp"

// mask_geometry_accel: 单件掩码 ROI 的清理与几何统计核（方案 5.1）
// 输入: 8位掩码像素（0/1，非零视为前景），AXIS 流，ROI 内按行序
// 输出: 同尺寸清理掩码 AXIS 流 + AXI-Lite 统计寄存器（一次完成快照）
// 统计: m00/m10/m01/m20/m02/m11（二阶矩，64位）+ min/max 行列
// 可旁路: erode_enable / dilate_enable 分别控制 3x3 腐蚀/膨胀（都为 0 则透传）
// 设计目标: 像素主循环 II=1；实际以综合/协同仿真报告为准

typedef ap_axiu<32,0,0,0> axis_word;
typedef ap_uint<64> stat_t;

#define MASK_GEOM_MAX_W 640
#define MASK_GEOM_MAX_H 480

// 状态寄存器编码
#define MASK_GEOM_STATUS_DONE         0x1u  // 新结果就绪
#define MASK_GEOM_STATUS_OVERFLOW     0x2u  // 任一统计量 64 位回绕
#define MASK_GEOM_STATUS_FORMAT_ERROR 0x4u  // 尺寸非法/超限

// 二值规范化：任意像素 -> 0/1
static void binarize_stage(hls::stream<axis_word>& input, hls::stream<unsigned char>& mask, unsigned n){
#pragma HLS INLINE off
    for(unsigned i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        mask.write((input.read().data & 0xff)!=0 ? 1 : 0);
    }
}

// 可旁路腐蚀 stage（复用 morph_stream 的 3x3 形态学，边界补零）
static void erode_stage(hls::stream<unsigned char>& in, hls::stream<unsigned char>& out,
                        unsigned width, unsigned height, bool enable){
#pragma HLS INLINE off
    if(enable){
        morph3<true>(in,out,width,height);
    } else {
        for(unsigned i=0;i<width*height;i++){
#pragma HLS PIPELINE II=1
            out.write(in.read());
        }
    }
}

// 可旁路膨胀 stage
static void dilate_stage(hls::stream<unsigned char>& in, hls::stream<unsigned char>& out,
                        unsigned width, unsigned height, bool enable){
#pragma HLS INLINE off
    if(enable){
        morph3<false>(in,out,width,height);
    } else {
        for(unsigned i=0;i<width*height;i++){
#pragma HLS PIPELINE II=1
            out.write(in.read());
        }
    }
}

// 统计 + 输出清理掩码：ROI 内坐标累加二阶矩，边界已由形态学补零
static void stats_and_pack_stage(hls::stream<unsigned char>& mask, hls::stream<axis_word>& output,
        unsigned width, unsigned height,
        stat_t* m00, stat_t* m10, stat_t* m01, stat_t* m20, stat_t* m02, stat_t* m11,
        unsigned* min_row, unsigned* max_row, unsigned* min_col, unsigned* max_col, bool* overflow){
#pragma HLS INLINE off
    ap_uint<64> a_m00=0, a_m10=0, a_m01=0, a_m20=0, a_m02=0, a_m11=0;
    unsigned rmin=height, rmax=0, cmin=width, cmax=0;
    bool ovf=false;
    for(unsigned y=0;y<height;y++){
        for(unsigned x=0;x<width;x++){
#pragma HLS PIPELINE II=1
            unsigned char v = mask.read();
            axis_word w; w.data = v; w.keep = 15; w.strb = 15;
            w.last = ((y==height-1) && (x==width-1)) ? 1 : 0;
            output.write(w);
            if(v!=0){
                ap_uint<64> ux = (ap_uint<64>)(ap_uint<32>)x;
                ap_uint<64> uy = (ap_uint<64>)(ap_uint<32>)y;
                ap_uint<64> t;
                t = a_m00 + 1;        if(t<a_m00) ovf=true; a_m00 = t;
                t = a_m10 + ux;       if(t<a_m10) ovf=true; a_m10 = t;
                t = a_m01 + uy;       if(t<a_m01) ovf=true; a_m01 = t;
                t = a_m20 + ux*ux;    if(t<a_m20) ovf=true; a_m20 = t;
                t = a_m02 + uy*uy;    if(t<a_m02) ovf=true; a_m02 = t;
                t = a_m11 + ux*uy;    if(t<a_m11) ovf=true; a_m11 = t;
                if(x<cmin) cmin=x; if(x>cmax) cmax=x;
                if(y<rmin) rmin=y;   if(y>rmax) rmax=y;
            }
        }
    }
    // 空掩码约定：行列边界统一输出 0，PS 结合 m00==0 判定为空
    if(a_m00==0){ rmin=0; rmax=0; cmin=0; cmax=0; }
    *m00=a_m00; *m10=a_m10; *m01=a_m01; *m20=a_m20; *m02=a_m02; *m11=a_m11;
    *min_row=rmin; *max_row=rmax; *min_col=cmin; *max_col=cmax;
    *overflow=ovf;
}

// Keep validation and register packing OUTSIDE the canonical dataflow region.
// Each stream has exactly one producer and consumer; stages must not be inlined.
static void geometry_pipeline(
    hls::stream<axis_word>& input, hls::stream<axis_word>& output,
    unsigned width, unsigned height, unsigned count, bool erode, bool dilate,
    stat_t& m00, stat_t& m10, stat_t& m01, stat_t& m20, stat_t& m02, stat_t& m11,
    unsigned& rmin, unsigned& rmax, unsigned& cmin, unsigned& cmax, bool& ovf){
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<unsigned char> mask("mask"), ero("ero"), opn("opn");
#pragma HLS STREAM variable=mask depth=64
#pragma HLS STREAM variable=ero depth=64
#pragma HLS STREAM variable=opn depth=64
    binarize_stage(input, mask, count);
    erode_stage(mask, ero, width, height, erode);
    dilate_stage(ero, opn, width, height, dilate);
    stats_and_pack_stage(opn, output, width, height,
                         &m00, &m10, &m01, &m20, &m02, &m11,
                         &rmin, &rmax, &cmin, &cmax, &ovf);
}

// 顶层：AXIS 掩码流进/出，统计量经 AXI-Lite 结果寄存器（64 位拆高/低读取，同一快照）
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
    unsigned& status){
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE s_axilite port=width         bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=height        bundle=control offset=0x18
#pragma HLS INTERFACE s_axilite port=erode_enable  bundle=control offset=0x20
#pragma HLS INTERFACE s_axilite port=dilate_enable bundle=control offset=0x28
#pragma HLS INTERFACE s_axilite port=job_id        bundle=control offset=0x30
#pragma HLS INTERFACE s_axilite port=m00_lo bundle=control offset=0x38
#pragma HLS INTERFACE s_axilite port=m00_hi bundle=control offset=0x40
#pragma HLS INTERFACE s_axilite port=m10_lo bundle=control offset=0x48
#pragma HLS INTERFACE s_axilite port=m10_hi bundle=control offset=0x50
#pragma HLS INTERFACE s_axilite port=m01_lo bundle=control offset=0x58
#pragma HLS INTERFACE s_axilite port=m01_hi bundle=control offset=0x60
#pragma HLS INTERFACE s_axilite port=m20_lo bundle=control offset=0x68
#pragma HLS INTERFACE s_axilite port=m20_hi bundle=control offset=0x70
#pragma HLS INTERFACE s_axilite port=m02_lo bundle=control offset=0x78
#pragma HLS INTERFACE s_axilite port=m02_hi bundle=control offset=0x80
#pragma HLS INTERFACE s_axilite port=m11_lo bundle=control offset=0x88
#pragma HLS INTERFACE s_axilite port=m11_hi bundle=control offset=0x90
#pragma HLS INTERFACE s_axilite port=min_row bundle=control offset=0x98
#pragma HLS INTERFACE s_axilite port=max_row bundle=control offset=0xA0
#pragma HLS INTERFACE s_axilite port=min_col bundle=control offset=0xA8
#pragma HLS INTERFACE s_axilite port=max_col bundle=control offset=0xB0
#pragma HLS INTERFACE s_axilite port=status  bundle=control offset=0xB8
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    // 格式检查：空尺寸或超限不处理，直接置错误状态
    if(width==0 || height==0 || width>MASK_GEOM_MAX_W || height>MASK_GEOM_MAX_H){
        status = MASK_GEOM_STATUS_FORMAT_ERROR;
        m00_lo=0; m00_hi=0; m10_lo=0; m10_hi=0;
        m01_lo=0; m01_hi=0; m20_lo=0; m20_hi=0;
        m02_lo=0; m02_hi=0; m11_lo=0; m11_hi=0;
        min_row=0; max_row=0; min_col=0; max_col=0;
        return;
    }

    stat_t m00,m10,m01,m20,m02,m11;
    unsigned rmin,rmax,cmin,cmax;
    bool ovf;
    geometry_pipeline(input, output, width, height, width*height,
                      erode_enable!=0, dilate_enable!=0,
                      m00,m10,m01,m20,m02,m11,rmin,rmax,cmin,cmax,ovf);

    // 结果统一锁存：64 位统计拆高/低，PS 读同一快照
    m00_lo = (ap_uint<32>)m00(31,0);  m00_hi = (ap_uint<32>)m00(63,32);
    m10_lo = (ap_uint<32>)m10(31,0);  m10_hi = (ap_uint<32>)m10(63,32);
    m01_lo = (ap_uint<32>)m01(31,0);  m01_hi = (ap_uint<32>)m01(63,32);
    m20_lo = (ap_uint<32>)m20(31,0);  m20_hi = (ap_uint<32>)m20(63,32);
    m02_lo = (ap_uint<32>)m02(31,0);  m02_hi = (ap_uint<32>)m02(63,32);
    m11_lo = (ap_uint<32>)m11(31,0);  m11_hi = (ap_uint<32>)m11(63,32);
    min_row=rmin; max_row=rmax; min_col=cmin; max_col=cmax;
    status = MASK_GEOM_STATUS_DONE | (ovf ? MASK_GEOM_STATUS_OVERFLOW : 0);
}
