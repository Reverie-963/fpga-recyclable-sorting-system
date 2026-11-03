/* Mathematical front-end for the next FPGA candidate, not a deployed backend.
 * Generates the current 5-coefficient, 5-iteration normalized pixel rays.
 * No HLS scheduling/resource/RTL or hardware performance claim is made here.
 */
#include <math.h>

/* p = fx,fy,cx,cy,k1,k2,p1,p2,k3. xy is row-major interleaved double. */
int inverse_ray_model(unsigned width, unsigned height, const double p[9], double *xy) {
    if (!width || width>640 || !height || height>480) return 4;
    for (unsigned i=0;i<9;++i) if (!isfinite(p[i])) return 4;
    if (!(p[0]>0.) || !(p[1]>0.)) return 4;
    const double ifx=1./p[0], ify=1./p[1];
    for (unsigned v=0;v<height;++v) {
        for (unsigned u=0;u<width;++u) {
            const double x0=(u-p[2])*ifx, y0=(v-p[3])*ify;
            double x=x0, y=y0;
            for (unsigned iteration=0;iteration<5;++iteration) {
                double r2=x*x+y*y;
                double reciprocal=1./(1.+((p[8]*r2+p[5])*r2+p[4])*r2);
                if (reciprocal<0.) { x=x0;y=y0;break; }
                double dx=2.*p[6]*x*y+p[7]*(r2+2.*x*x);
                double dy=p[6]*(r2+2.*y*y)+2.*p[7]*x*y;
                x=(x0-dx)*reciprocal;
                y=(y0-dy)*reciprocal;
            }
            if (!isfinite(x) || !isfinite(y)) return 2;
            unsigned index=v*width+u;
            xy[2*index]=x;xy[2*index+1]=y;
        }
    }
    return 0;
}
