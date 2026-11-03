"""visual_evidence_accel PYNQ overlay 驱动（方案 5.2）。

在 PYNQ 板上运行（需 pynq 库）。双核 bit/hwh 由 integrate_bd.tcl 构建，
复制到 board/vision.bit + board/vision.hwh，与 driver 一并传到板端。

用法:
    from km1_visual_overlay import VisualAccel
    vis = VisualAccel('/path/to/vision.bit')
    stats, out = vis.run(frameA, frameB, threshold=30, mode=0, job_id=3)
    # frameA/B: (H,W) uint8（灰度 0-255 或掩码 0/1）
    # stats: mode=1 -> inter/erased/beyond/union(64bit) + 变化 bbox + status
    #         mode=0 -> diff_count(64bit) + 变化 bbox + status
    # out: 核输出的变化掩码流（uint32 每像素，低字节 0/1）

寄存器偏移与综合报告核对一致（s_axi_control 基地址 0x40010000 等，从 hwh 解析）。
统计在完成时锁存，DONE 后读即同一次快照。
"""

from km1_dma_runtime import DualCoreSession, image_u8, uint_arg

# --- visual_evidence_accel AXI-Lite 偏移 ---
CTRL = 0x00            # 0=AP_START 1=AP_DONE 2=AP_IDLE
WIDTH = 0x10
HEIGHT = 0x18
THRESHOLD = 0x20
MODE = 0x28
JOB_ID = 0x30
S0_LO, S0_HI = 0x38, 0x40
S1_LO, S1_HI = 0x48, 0x50
S2_LO, S2_HI = 0x58, 0x60
S3_LO, S3_HI = 0x68, 0x70
MIN_ROW, MAX_ROW = 0x78, 0x80
MIN_COL, MAX_COL = 0x88, 0x90
STATUS = 0x98

STAT_DONE = 0x1
STAT_OVERFLOW = 0x2
STAT_FORMAT_ERROR = 0x4


class VisualAccel:
    """封装双核 overlay 中的 visual 核：dma_vis_a(inputA/out) + dma_vis_b(inputB)。"""

    def __init__(self, bit_path='vision.bit', session=None):
        self.session = session if session is not None else DualCoreSession(bit_path)

    def run(self, frame_a, frame_b, threshold=30, mode=0, job_id=0, timeout_sec=5.0):
        """Return latched statistics and an owned flat uint32 NumPy array."""
        a, b = image_u8(frame_a), image_u8(frame_b)
        if a.shape != b.shape:
            raise ValueError('frame shapes must match')
        mode = uint_arg(mode, 'mode', 1)
        h, w = a.shape
        registers = {WIDTH:w, HEIGHT:h, THRESHOLD:uint_arg(threshold, 'threshold', 255),
                     MODE:mode, JOB_ID:uint_arg(job_id, 'job_id')}
        return self.session.run('visual_0', [('dma_vis_a', a), ('dma_vis_b', b)],
                                'dma_vis_a', registers,
                                lambda reg: self._read_stats(reg, mode), timeout_sec)

    def _read_stats(self, reg, mode):
        def r64(lo, hi):
            # Exclusive session: no new job can alter this completed snapshot.
            return (reg.read(hi) << 32) | reg.read(lo)
        base = {
            'min_row': reg.read(MIN_ROW),
            'max_row': reg.read(MAX_ROW),
            'min_col': reg.read(MIN_COL),
            'max_col': reg.read(MAX_COL),
            'job_id': reg.read(JOB_ID),
            'status': reg.read(STATUS),
            'ctrl': reg.read(CTRL),
        }
        if mode == 0:
            base['diff_count'] = r64(S0_LO, S0_HI)
        else:
            base['inter'] = r64(S0_LO, S0_HI)
            base['erased'] = r64(S1_LO, S1_HI)
            base['beyond'] = r64(S2_LO, S2_HI)
            base['union'] = r64(S3_LO, S3_HI)
        return base
