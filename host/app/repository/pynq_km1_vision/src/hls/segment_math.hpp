#ifndef SEGMENT_MATH_HPP
#define SEGMENT_MATH_HPP
inline unsigned segment_pixel(unsigned p,unsigned bg,unsigned threshold) {
    int b=int(p&255)-int(bg&255);
    int g=int((p>>8)&255)-int((bg>>8)&255);
    int r=int((p>>16)&255)-int((bg>>16)&255);
    unsigned distance=(b<0?-b:b)+(g<0?-g:g)+(r<0?-r:r);
    return distance>threshold?255:0;
}
#endif
