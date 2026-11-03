#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

// visual_evidence_accel: 视觉核验统计核（方案 5.2）
// 输入: 同一固定视角、同一 ROI 的前后两路像素（A=前帧/前态，B=后帧/后态），
//       AXIS 流成对送入，逐像素同步比较
// mode=0 灰度: 统计绝对差分超过 threshold 的像素数，包围框=这些变化像素
// mode=1 掩码: a/b 非零视为前景，统计交集/消失/新增/并集面积，包围框=变化(a!=b)像素
// 输出: 变化掩码 AXIS 流（change 像素=1，其余 0），供 PS 可视化/进一步核验
// 结果: 4 个 64 位统计（AXI-Lite 拆高/低，同一快照）+ 变化包围框 min/max 行列 + status
//   stat0: 灰度=差分像素数 ; 掩码=交集面积(inter)
//   stat1: 灰度=0         ; 掩码=消失面积(erased, A&~B)
//   stat2: 灰度=0         ; 掩码=新增面积(beyond, ~A&B)
//   stat3: 灰度=0         ; 掩码=并集面积(union, A|B)
// 约定: 无变化像素时包围框统一输出 0；空掩码判定由 PS 结合 stat0 解释
// 设计目标: 像素主循环 II=1；实际以综合/协同仿真报告为准

typedef ap_axiu<32,0,0,0> axis_word;
typedef ap_uint<64> stat_t;

#define VIS_EVID_MAX_W 640
#define VIS_EVID_MAX_H 480

// 状态寄存器编码
#define VIS_EVID_STATUS_DONE         0x1u  // 新结果就绪
#define VIS_EVID_STATUS_OVERFLOW     0x2u  // 任一统计量 64 位回绕
#define VIS_EVID_STATUS_FORMAT_ERROR 0x4u  // 尺寸非法/超限

void visual_evidence_accel(
    hls::stream<axis_word>& inputA, hls::stream<axis_word>& inputB, hls::stream<axis_word>& output,
    unsigned width, unsigned height, unsigned threshold, unsigned mode, unsigned job_id,
    ap_uint<32>& stat0_lo, ap_uint<32>& stat0_hi,
    ap_uint<32>& stat1_lo, ap_uint<32>& stat1_hi,
    ap_uint<32>& stat2_lo, ap_uint<32>& stat2_hi,
    ap_uint<32>& stat3_lo, ap_uint<32>& stat3_hi,
    unsigned& min_row, unsigned& max_row, unsigned& min_col, unsigned& max_col,
    unsigned& status){
#pragma HLS INTERFACE axis port=inputA
#pragma HLS INTERFACE axis port=inputB
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE s_axilite port=width     bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=height    bundle=control offset=0x18
#pragma HLS INTERFACE s_axilite port=threshold bundle=control offset=0x20
#pragma HLS INTERFACE s_axilite port=mode      bundle=control offset=0x28
#pragma HLS INTERFACE s_axilite port=job_id    bundle=control offset=0x30
#pragma HLS INTERFACE s_axilite port=stat0_lo  bundle=control offset=0x38
#pragma HLS INTERFACE s_axilite port=stat0_hi  bundle=control offset=0x40
#pragma HLS INTERFACE s_axilite port=stat1_lo  bundle=control offset=0x48
#pragma HLS INTERFACE s_axilite port=stat1_hi  bundle=control offset=0x50
#pragma HLS INTERFACE s_axilite port=stat2_lo  bundle=control offset=0x58
#pragma HLS INTERFACE s_axilite port=stat2_hi  bundle=control offset=0x60
#pragma HLS INTERFACE s_axilite port=stat3_lo  bundle=control offset=0x68
#pragma HLS INTERFACE s_axilite port=stat3_hi  bundle=control offset=0x70
#pragma HLS INTERFACE s_axilite port=min_row   bundle=control offset=0x78
#pragma HLS INTERFACE s_axilite port=max_row   bundle=control offset=0x80
#pragma HLS INTERFACE s_axilite port=min_col   bundle=control offset=0x88
#pragma HLS INTERFACE s_axilite port=max_col   bundle=control offset=0x90
#pragma HLS INTERFACE s_axilite port=status    bundle=control offset=0x98
#pragma HLS INTERFACE s_axilite port=return    bundle=control

    // 格式检查：空尺寸或超限不处理，直接置错误状态
    if(width==0 || height==0 || width>VIS_EVID_MAX_W || height>VIS_EVID_MAX_H){
        status = VIS_EVID_STATUS_FORMAT_ERROR;
        stat0_lo=0; stat0_hi=0; stat1_lo=0; stat1_hi=0;
        stat2_lo=0; stat2_hi=0; stat3_lo=0; stat3_hi=0;
        min_row=0; max_row=0; min_col=0; max_col=0;
        return;
    }

    stat_t s0=0, s1=0, s2=0, s3=0;
    unsigned rmin=height, rmax=0, cmin=width, cmax=0;
    bool ovf=false, any_change=false;
    const unsigned n = width*height;
    const bool mask_mode = (mode!=0);

    for(unsigned i=0;i<n;i++){
#pragma HLS PIPELINE II=1
        unsigned char a = (unsigned char)inputA.read().data;
        unsigned char b = (unsigned char)inputB.read().data;
        unsigned y = i / width;
        unsigned x = i % width;

        bool change;
        unsigned char ov;
        if(!mask_mode){
            unsigned char diff = (a>b) ? (a-b) : (b-a);
            change = (diff > (unsigned char)threshold);
            if(change){ stat_t t = s0 + 1; if(t<s0) ovf=true; s0=t; }
            ov = change ? 1 : 0;
        } else {
            unsigned av = (a!=0) ? 1 : 0;
            unsigned bv = (b!=0) ? 1 : 0;
            stat_t t;
            t = s3 + (av|bv);    if(t<s3) ovf=true; s3 = t;   // union
            t = s2 + (bv & ~av); if(t<s2) ovf=true; s2 = t;   // beyond
            t = s1 + (av & ~bv); if(t<s1) ovf=true; s1 = t;   // erased
            t = s0 + (av & bv);  if(t<s0) ovf=true; s0 = t;   // inter
            change = (av!=bv);
            ov = change ? 1 : 0;
        }

        axis_word w; w.data = ov; w.keep = 15; w.strb = 15;
        w.last = (i==n-1) ? 1 : 0;
        output.write(w);

        if(change){
            any_change = true;
            if(x<cmin) cmin=x; if(x>cmax) cmax=x;
            if(y<rmin) rmin=y; if(y>rmax) rmax=y;
        }
    }

    // 无变化像素约定：包围框统一输出 0
    if(!any_change){ rmin=0; rmax=0; cmin=0; cmax=0; }

    stat0_lo=(ap_uint<32>)s0(31,0);  stat0_hi=(ap_uint<32>)s0(63,32);
    stat1_lo=(ap_uint<32>)s1(31,0);  stat1_hi=(ap_uint<32>)s1(63,32);
    stat2_lo=(ap_uint<32>)s2(31,0);  stat2_hi=(ap_uint<32>)s2(63,32);
    stat3_lo=(ap_uint<32>)s3(31,0);  stat3_hi=(ap_uint<32>)s3(63,32);
    min_row=rmin; max_row=rmax; min_col=cmin; max_col=cmax;
    status = VIS_EVID_STATUS_DONE | (ovf ? VIS_EVID_STATUS_OVERFLOW : 0);
}
