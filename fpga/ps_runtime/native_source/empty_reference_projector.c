/* Forward pinhole projection only: no calibration solver or Jacobian output.
 * Brown-Conrady/rational model, coefficients in OpenCV's documented order.
 * Compile without fast-math or FP contraction; the remap uses float32 maps.
 * This is a CPU helper and mathematical baseline for future PL work.
 */
#include <stddef.h>

int empty_reference_project(unsigned count, const double *points,
                            const double *rotation, const double *translation,
                            const double *intrinsic, const double *coefficients,
                            unsigned coefficient_count, double *pixels) {
    if (!points || !rotation || !translation || !intrinsic || !coefficients || !pixels ||
        count > 640u*480u || !(coefficient_count == 4 || coefficient_count == 5 ||
                             coefficient_count == 8 || coefficient_count == 12)) return -1;
    double lens[12] = {0};
    for (unsigned j = 0; j < coefficient_count; ++j) lens[j] = coefficients[j];
    for (unsigned i = 0; i < count; ++i) {
        const double *world = points+3*i;
        double camera[3];
        for (unsigned row = 0; row < 3; ++row) {
            const double *matrix_row = rotation+3*row;
            camera[row] = matrix_row[0]*world[0]+matrix_row[1]*world[1]+
                          matrix_row[2]*world[2]+translation[row];
        }
        double inverse_depth = camera[2] != 0 ? 1./camera[2] : 1.;
        double x = camera[0]*inverse_depth, y = camera[1]*inverse_depth;
        double radius2 = x*x+y*y, radius4 = radius2*radius2, radius6 = radius4*radius2;
        double radial_top = 1+ lens[0]*radius2+ lens[1]*radius4+ lens[4]*radius6;
        double radial_bottom_inverse = 1./(1+ lens[5]*radius2+ lens[6]*radius4+ lens[7]*radius6);
        double twice_xy = 2*x*y;
        double distorted_x = x*radial_top*radial_bottom_inverse + lens[2]*twice_xy +
            lens[3]*(radius2+2*x*x) + lens[8]*radius2 + lens[9]*radius4;
        double distorted_y = y*radial_top*radial_bottom_inverse + lens[2]*(radius2+2*y*y) +
            lens[3]*twice_xy + lens[10]*radius2 + lens[11]*radius4;
        pixels[2*i] = distorted_x*intrinsic[0]+intrinsic[2];
        pixels[2*i+1] = distorted_y*intrinsic[4]+intrinsic[5];
    }
    return 0;
}
