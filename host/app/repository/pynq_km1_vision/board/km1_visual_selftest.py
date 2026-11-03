"""visual_evidence_accel 板上自测：构造已知差异的两帧，核对统计与变化包围框。

在 PYNQ 板端执行：python km1_visual_selftest.py
预期：
  mask 模式: 3x3 块 A(中心1x1) -> B(中心1x2)：inter=1, erased=0, beyond=1, union=2, bbox 覆盖变化列
  灰度模式: 单点差分超阈值 -> diff_count=1
"""

import numpy as np
from km1_visual_overlay import VisualAccel


def _chk(name, got, exp):
    ok = got == exp
    print('%s %-24s got=%s exp=%s' % ('PASS' if ok else 'FAIL', name, got, exp))
    return ok


def main():
    vis = VisualAccel('vision.bit')
    ok = True

    # 用例1：掩码模式，A 中心单点 -> B 中心横向两点
    a = np.zeros((4, 5), np.uint8); a[2, 2] = 1
    b = np.zeros((4, 5), np.uint8); b[2, 2] = 1; b[2, 3] = 1
    s, _ = vis.run(a, b, threshold=0, mode=1, job_id=9)
    ok &= _chk('inter(交集)', s['inter'], 1)
    ok &= _chk('erased(消失)', s['erased'], 0)
    ok &= _chk('beyond(新增)', s['beyond'], 1)
    ok &= _chk('union(并集)', s['union'], 2)
    ok &= _chk('bbox_row', (s['min_row'], s['max_row']), (2, 2))
    ok &= _chk('bbox_col', (s['min_col'], s['max_col']), (3, 3))  # 变化像素仅 (2,3)，不含共同前景 (2,2)
    ok &= _chk('job_id', s['job_id'], 9)
    ok &= _chk('status_DONE', s['status'] & 0x1, 0x1)

    # 用例2：掩码模式，A->B 整体右移（交集+消失+新增都有）
    a = np.zeros((4, 5), np.uint8); a[1, 1] = 1; a[1, 2] = 1
    b = np.zeros((4, 5), np.uint8); b[1, 2] = 1; b[1, 3] = 1
    s2, _ = vis.run(a, b, threshold=0, mode=1, job_id=10)
    ok &= _chk('shift_inter', s2['inter'], 1)
    ok &= _chk('shift_erased', s2['erased'], 1)
    ok &= _chk('shift_beyond', s2['beyond'], 1)
    ok &= _chk('shift_union', s2['union'], 3)
    # 变化像素 = erased(1,1) ∪ beyond(1,3) -> 变化闭区间 row[1,1] col[1,3]（不含共同前景 (1,2)）
    ok &= _chk('shift_bbox', (s2['min_row'], s2['max_row'], s2['min_col'], s2['max_col']), (1, 1, 1, 3))

    # 用例3：灰度模式，单点差 40 > threshold 30
    ga = np.full((4, 5), 50, np.uint8)
    gb = ga.copy(); gb[2, 2] = 90
    s3, _ = vis.run(ga, gb, threshold=30, mode=0, job_id=11)
    ok &= _chk('gray_diff_count', s3['diff_count'], 1)
    ok &= _chk('gray_bbox', (s3['min_row'], s3['min_col']), (2, 2))

    # 用例4：灰度模式，全等帧 -> 无变化，bbox 归零
    ga2 = np.full((4, 5), 60, np.uint8); gb2 = ga2.copy()
    s4, _ = vis.run(ga2, gb2, threshold=30, mode=0, job_id=12)
    ok &= _chk('gray_same_diff', s4['diff_count'], 0)
    ok &= _chk('gray_same_bbox', (s4['min_row'], s4['max_row'], s4['min_col'], s4['max_col']), (0, 0, 0, 0))

    print('=== visual 板端自测%s === ok=%d' % ('全部通过' if ok else '有失败', ok))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
