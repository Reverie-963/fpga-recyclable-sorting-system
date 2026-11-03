#include <stdint.h>
#include <hls_stream.h>

// Pixel payload: gray[7:0], smooth[15:8], edges[23:16], raw change[24],
// persistent[25], eroded/opened[26], RGB absolute difference[36:27].
typedef uint64_t Pixel;
static const unsigned MAX_PIXELS = 640 * 480;

static void read_image(const uint32_t *memory, hls::stream<uint32_t> &out, unsigned count) {
#pragma HLS INLINE off
    for (unsigned i=0; i<count; ++i) {
#pragma HLS PIPELINE II=1
        out.write(memory[i]);
    }
}

static void write_words(hls::stream<uint32_t> &in, uint32_t *memory, unsigned count) {
#pragma HLS INLINE off
    for (unsigned i=0; i<count; ++i) {
#pragma HLS PIPELINE II=1
        memory[i] = in.read();
    }
}

static void rgb_gray(hls::stream<uint32_t> &base, hls::stream<uint32_t> &frame,
        hls::stream<uint32_t> &old_history, hls::stream<uint32_t> &new_history,
        hls::stream<Pixel> &out, unsigned count, unsigned threshold, unsigned reset) {
#pragma HLS INLINE off
    uint32_t prior=0, next=0;
    for (unsigned i=0; i<count; ++i) {
#pragma HLS PIPELINE II=1
        if ((i & 31) == 0) { prior=old_history.read(); next=0; }
        uint32_t a=base.read(), b=frame.read();
        unsigned delta=0;
        for (unsigned k=0; k<3; ++k) {
#pragma HLS UNROLL
            unsigned av=(a>>(8*k))&255, bv=(b>>(8*k))&255;
            delta += av>bv ? av-bv : bv-av;
        }
        unsigned gray=(29*(b&255)+150*((b>>8)&255)+77*((b>>16)&255)+128)>>8;
        bool changed=delta>threshold;
        bool persistent=changed && (reset || ((prior>>(i&31))&1));
        next |= uint32_t(persistent) << (i&31);
        out.write(Pixel(gray) | (Pixel(changed)<<24) | (Pixel(persistent)<<25) | (Pixel(delta)<<27));
        if ((i&31)==31 || i+1==count) new_history.write(next);
    }
}

// A 3x3 stencil uses two stored rows and a 3x3 shift window.
// KRAD=1, NLB=2, WARMUP=width+1, TOTAL_ITER=width*height+WARMUP.
// All stages drain with inline zero input and produce exactly width*height tokens.
template<int MODE>
static void stencil(hls::stream<Pixel> &in, hls::stream<Pixel> &out,
        unsigned width, unsigned height) {
#pragma HLS INLINE off
    Pixel rows[2][640];
#pragma HLS ARRAY_PARTITION variable=rows complete dim=1
#pragma HLS BIND_STORAGE variable=rows type=ram_t2p impl=bram
    Pixel window[3][3] = {};
#pragma HLS ARRAY_PARTITION variable=window complete dim=0
    const unsigned count=width*height;
    const unsigned warmup=width+1;
    const unsigned total_iter=count+warmup;
    unsigned col=0, ox=0, oy=0;
    for (unsigned n=0; n<total_iter; ++n) {
#pragma HLS PIPELINE II=1
        Pixel current = n<count ? in.read() : Pixel(0);
        Pixel middle = n>=width ? rows[0][col] : Pixel(0);
        Pixel top = n>=2*width ? rows[1][col] : Pixel(0);
        rows[1][col]=middle; rows[0][col]=current;
        for (unsigned y=0; y<3; ++y) {
#pragma HLS UNROLL
            window[y][0]=window[y][1]; window[y][1]=window[y][2];
        }
        window[0][2]=top; window[1][2]=middle; window[2][2]=current;
        if (n>=warmup) {
            int v[3][3];
#pragma HLS ARRAY_PARTITION variable=v complete dim=0
            for (unsigned y=0; y<3; ++y) {
#pragma HLS UNROLL
                for (unsigned x=0; x<3; ++x) {
#pragma HLS UNROLL
                    bool valid=int(ox)+int(x)-1>=0 && int(ox)+int(x)-1<int(width)
                        && int(oy)+int(y)-1>=0 && int(oy)+int(y)-1<int(height);
                    unsigned shift=MODE==0 ? 0 : MODE==1 ? 8 : MODE==2 ? 24 : 26;
                    unsigned mask=MODE<2 ? 255 : 1;
                    v[y][x]=valid ? int((window[y][x]>>shift)&mask) : 0;
                }
            }
            Pixel center=window[1][1];
            if (MODE==0) {
                unsigned sum=0;
                for (unsigned y=0;y<3;++y) {
#pragma HLS UNROLL
                    for (unsigned x=0;x<3;++x) {
#pragma HLS UNROLL
                        sum+=v[y][x];
                    }
                }
                center |= Pixel((sum+4)/9)<<8;
            } else if (MODE==1) {
                int gx=-v[0][0]+v[0][2]-2*v[1][0]+2*v[1][2]-v[2][0]+v[2][2];
                int gy=-v[0][0]-2*v[0][1]-v[0][2]+v[2][0]+2*v[2][1]+v[2][2];
                unsigned magnitude=(gx<0?-gx:gx)+(gy<0?-gy:gy);
                center |= Pixel(magnitude>255?255:magnitude)<<16;
            } else {
                unsigned value=MODE==2 ? 1 : 0;
                for (unsigned y=0;y<3;++y) {
#pragma HLS UNROLL
                    for (unsigned x=0;x<3;++x) {
#pragma HLS UNROLL
                        if (MODE==2) value &= v[y][x]; else value |= v[y][x];
                    }
                }
                center=(center & ~(Pixel(1)<<26)) | (Pixel(value)<<26);
            }
            out.write(center);
            if (++ox==width) { ox=0; ++oy; }
        }
        if (++col==width) col=0;
    }
}

static void summarize(hls::stream<Pixel> &in, hls::stream<uint32_t> &images,
        hls::stream<uint32_t> &results, const uint32_t rect[6][4],
        unsigned width, unsigned height) {
#pragma HLS INLINE off
    uint32_t counters[6][8], extra[6][4], histogram[16];
#pragma HLS ARRAY_PARTITION variable=counters complete dim=0
#pragma HLS ARRAY_PARTITION variable=extra complete dim=0
#pragma HLS ARRAY_PARTITION variable=histogram complete dim=0
    for(unsigned r=0;r<6;++r) {
        for(unsigned k=0;k<8;++k) counters[r][k]=0;
        for(unsigned k=0;k<4;++k) extra[r][k]=0;
        counters[r][3]=width; counters[r][4]=height;
        counters[r][7]=(rect[r][2]-rect[r][0])*(rect[r][3]-rect[r][1]);
    }
    for(unsigned k=0;k<16;++k) histogram[k]=0;
    const unsigned count=width*height;
    unsigned x=0, y=0;
    for(unsigned i=0;i<count;++i) {
#pragma HLS PIPELINE II=1
        Pixel value=in.read();
        unsigned gray=value&255, smooth=(value>>8)&255, edge=(value>>16)&255;
        bool changed=(value>>24)&1, persistent=(value>>25)&1, clean=(value>>26)&1;
        unsigned delta=(value>>27)&1023;
        for(unsigned k=0;k<16;++k) {
#pragma HLS UNROLL
            histogram[k] += (gray>>4)==k;
        }
        for(unsigned r=0;r<6;++r) {
#pragma HLS UNROLL
            if(x>=rect[r][0] && x<rect[r][2] && y>=rect[r][1] && y<rect[r][3]) {
                counters[r][0]+=changed; counters[r][1]+=persistent; counters[r][2]+=delta;
                if(changed) {
                    if(x<counters[r][3]) counters[r][3]=x;
                    if(y<counters[r][4]) counters[r][4]=y;
                    if(x>counters[r][5]) counters[r][5]=x;
                    if(y>counters[r][6]) counters[r][6]=y;
                }
                extra[r][0]+=gray; extra[r][1]+=edge>=64;
                extra[r][2]+=clean; extra[r][3]+=smooth;
            }
        }
        images.write(gray | (smooth<<8) | (edge<<16) | (uint32_t(clean?255:0)<<24));
        if(++x==width) { x=0; ++y; }
    }
    for(unsigned r=0;r<6;++r) {
        if(!counters[r][0]) for(unsigned k=3;k<7;++k) counters[r][k]=0;
        for(unsigned k=0;k<8;++k) results.write(counters[r][k]);
    }
    for(unsigned k=0;k<16;++k) results.write(histogram[k]);
    for(unsigned r=0;r<6;++r) for(unsigned k=0;k<4;++k) results.write(extra[r][k]);
}

static void store_preview(hls::stream<uint32_t> &in, uint32_t *memory,
        unsigned count, unsigned enabled) {
#pragma HLS INLINE off
    if(enabled) {
        for(unsigned i=0;i<count;++i) {
#pragma HLS PIPELINE II=1
            memory[i]=in.read();
        }
    } else {
        for(unsigned i=0;i<count;++i) {
#pragma HLS PIPELINE II=1
            in.read();
        }
    }
}

static void pipeline(const uint32_t *baseline, const uint32_t *current,
        const uint32_t *prior_history, uint32_t *next_history, uint32_t *preview,
        uint32_t *results, const uint32_t rect[6][4], unsigned width, unsigned height,
        unsigned count, unsigned history_words, unsigned result_words,
        unsigned threshold, unsigned reset_history, unsigned preview_enabled) {
#pragma HLS DATAFLOW
    hls::stream<uint32_t> base, frame, prior, next, image, summary;
    hls::stream<Pixel> raw, smoothed, edged, eroded, opened;
#pragma HLS STREAM variable=base depth=64
#pragma HLS STREAM variable=frame depth=64
#pragma HLS STREAM variable=prior depth=64
#pragma HLS STREAM variable=next depth=64
#pragma HLS STREAM variable=image depth=64
#pragma HLS STREAM variable=summary depth=32
#pragma HLS STREAM variable=raw depth=64
#pragma HLS STREAM variable=smoothed depth=64
#pragma HLS STREAM variable=edged depth=64
#pragma HLS STREAM variable=eroded depth=64
#pragma HLS STREAM variable=opened depth=64
    read_image(baseline,base,count);
    read_image(current,frame,count);
    read_image(prior_history,prior,history_words);
    rgb_gray(base,frame,prior,next,raw,count,threshold,reset_history);
    stencil<0>(raw,smoothed,width,height);
    stencil<1>(smoothed,edged,width,height);
    stencil<2>(edged,eroded,width,height);
    stencil<3>(eroded,opened,width,height);
    summarize(opened,image,summary,rect,width,height);
    write_words(next,next_history,history_words);
    store_preview(image,preview,count,preview_enabled);
    write_words(summary,results,result_words);
}

extern "C" void station_frame_pipeline(const uint32_t *baseline, const uint32_t *current,
        const uint32_t *prior_history, uint32_t *next_history, uint32_t *preview,
        const uint32_t *rois, uint32_t *results, unsigned width, unsigned height,
        unsigned threshold, unsigned reset_history, unsigned preview_enabled,
        unsigned job_id, unsigned &completed_job, unsigned &status) {
#pragma HLS INTERFACE m_axi port=baseline offset=slave bundle=base_mem depth=307200 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=current offset=slave bundle=frame_mem depth=307200 max_read_burst_length=64 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=prior_history offset=slave bundle=old_hist_mem depth=9600 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=next_history offset=slave bundle=new_hist_mem depth=9600 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=preview offset=slave bundle=preview_mem depth=307200 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=rois offset=slave bundle=config_mem depth=24
#pragma HLS INTERFACE m_axi port=results offset=slave bundle=config_mem depth=88
#pragma HLS INTERFACE s_axilite port=baseline bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=current bundle=control offset=0x20
#pragma HLS INTERFACE s_axilite port=prior_history bundle=control offset=0x30
#pragma HLS INTERFACE s_axilite port=next_history bundle=control offset=0x40
#pragma HLS INTERFACE s_axilite port=preview bundle=control offset=0x50
#pragma HLS INTERFACE s_axilite port=rois bundle=control offset=0x60
#pragma HLS INTERFACE s_axilite port=results bundle=control offset=0x70
#pragma HLS INTERFACE s_axilite port=width bundle=control offset=0x80
#pragma HLS INTERFACE s_axilite port=height bundle=control offset=0x88
#pragma HLS INTERFACE s_axilite port=threshold bundle=control offset=0x90
#pragma HLS INTERFACE s_axilite port=reset_history bundle=control offset=0x98
#pragma HLS INTERFACE s_axilite port=preview_enabled bundle=control offset=0xa0
#pragma HLS INTERFACE s_axilite port=job_id bundle=control offset=0xa8
#pragma HLS INTERFACE s_axilite port=completed_job bundle=control offset=0xb0
#pragma HLS INTERFACE s_axilite port=status bundle=control offset=0xb8
#pragma HLS INTERFACE s_axilite port=return bundle=control
    status=4; completed_job=0;
    if(!width || !height || width>640 || height>480 || threshold>765 ||
            reset_history>1 || preview_enabled>1) return;
    uint32_t rect[6][4];
#pragma HLS ARRAY_PARTITION variable=rect complete dim=0
    for(unsigned r=0;r<6;++r) {
        for(unsigned k=0;k<4;++k) rect[r][k]=rois[r*4+k];
        if(rect[r][0]>=rect[r][2] || rect[r][1]>=rect[r][3] ||
                rect[r][2]>width || rect[r][3]>height) return;
    }
    unsigned count=width*height, history_words=(count+31)/32, result_words=88;
    pipeline(baseline,current,prior_history,next_history,preview,results,rect,width,height,
             count,history_words,result_words,threshold,reset_history,preview_enabled);
    completed_job=job_id; status=1;
}
