"""mask_geometry_accel 板上自测：构造已知掩码，核跑后核对面积/包围框/矩。

在 PYNQ 板端执行：python km1_mask_selftest.py
预期：3x3 中心 2 像素块 -> m00=2, bbox row[1,1] col[1,2]；全零 -> 空且 bbox 0。
"""

import numpy as np
from km1_mask_overlay import MaskAccel


def _chk(name, got, exp):
    ok = got == exp
    print('%s %-28s got=%s exp=%s' % ('PASS' if ok else 'FAIL', name, got, exp))
    return ok


def main():
    acc = MaskAccel('vision.bit')
    ok = True

    # 用例1：3x3 中心横向 2 像素
    m = np.zeros((3, 3), np.uint8)
    m[1, 1] = 1
    m[1, 2] = 1
    s, _ = acc.run(m, erode=0, dilate=0, job_id=42)
    ok &= _chk('m00(面积)', s['m00'], 2)
    ok &= _chk('m10(Σx)', s['m10'], 3)          # 1+2
    ok &= _chk('m01(Σy)', s['m01'], 2)          # 1+1
    ok &= _chk('bbox_row', (s['min_row'], s['max_row']), (1, 1))
    ok &= _chk('bbox_col', (s['min_col'], s['max_col']), (1, 2))
    ok &= _chk('job_id', s['job_id'], 42)
    ok &= _chk('status_DONE', s['status'] & 0x1, 0x1)

    # 用例2：全零 -> 空图，bbox 归零
    m0 = np.zeros((4, 5), np.uint8)
    s0, _ = acc.run(m0, erode=0, dilate=0, job_id=0)
    ok &= _chk('empty_m00', s0['m00'], 0)
    ok &= _chk('empty_bbox', (s0['min_row'], s0['max_row'], s0['min_col'], s0['max_col']), (0, 0, 0, 0))

    # 用例3：单点，核输出原始二阶矩 = Σx²/Σy²/Σxy（中心矩才为 0）
    m1 = np.zeros((5, 5), np.uint8)
    m1[2, 2] = 1
    s1, _ = acc.run(m1, erode=0, dilate=0, job_id=1)
    ok &= _chk('single_m20', s1['m20'], 4)   # x=2 -> x²=4
    ok &= _chk('single_m02', s1['m02'], 4)   # y=2 -> y²=4
    ok &= _chk('single_m11', s1['m11'], 4)   # x*y=4
    ok &= _chk('single_bbox', (s1['min_row'], s1['min_col']), (2, 2))

    print('=== 板端自测%s === ' % ('全部通过' if ok else '有失败'), 'ok=%d' % ok)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
