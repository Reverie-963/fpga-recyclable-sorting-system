"""mask_geometry_accel PYNQ overlay 驱动（方案 5.1）。

在 PYNQ 板上运行（需 pynq 库）。bit/hwh 由 tools/build_dual_core.ps1 构建生成，
复制到 board/vision.bit + board/vision.hwh，与 driver 一并传到板端。

用法:
    from km1_mask_overlay import MaskAccel
    acc = MaskAccel('/path/to/vision.bit')
    stats, out = acc.run(mask_2d, erode=1, dilate=0, job_id=7)
    # stats: m00..m11(二阶矩) + min/max 行列 + status
    # out: 核输出的清理掩码流（uint32 每像素，低字节为 0/1）

寄存器偏移与综合报告核对一致（s_axi_control 基地址 0x40000000）。
统计寄存器在完成时锁存（ap_vld），DONE 后读即同一次快照。
"""

from km1_dma_runtime import DualCoreSession, image_u8, uint_arg

# --- mask_geometry_accel AXI-Lite 偏移 ---
CTRL = 0x00          # 0=AP_START 1=AP_DONE 2=AP_IDLE 3=AP_READY
WIDTH = 0x10
HEIGHT = 0x18
ERODE = 0x20
DILATE = 0x28
JOB_ID = 0x30
M00_LO, M00_HI = 0x38, 0x40
M10_LO, M10_HI = 0x48, 0x50
M01_LO, M01_HI = 0x58, 0x60
M20_LO, M20_HI = 0x68, 0x70
M02_LO, M02_HI = 0x78, 0x80
M11_LO, M11_HI = 0x88, 0x90
MIN_ROW, MAX_ROW = 0x98, 0xA0
MIN_COL, MAX_COL = 0xA8, 0xB0
STATUS = 0xB8

# status 编码
STAT_DONE = 0x1
STAT_OVERFLOW = 0x2
STAT_FORMAT_ERROR = 0x4


class MaskAccel:
    """封装 PYNQ overlay：AXI DMA(MM2S→核 / S2MM←核) + mask 核 AXI-Lite 控制/统计。"""

    def __init__(self, bit_path='vision.bit', session=None):
        self.session = session if session is not None else DualCoreSession(bit_path)

    def run(self, mask, erode=0, dilate=0, job_id=0, timeout_sec=5.0):
        """Return latched statistics and an owned flat uint32 NumPy array."""
        mask = image_u8(mask)
        h, w = mask.shape
        registers = {WIDTH:w, HEIGHT:h, ERODE:uint_arg(erode, 'erode', 1),
                     DILATE:uint_arg(dilate, 'dilate', 1), JOB_ID:uint_arg(job_id, 'job_id')}
        return self.session.run('mask_0', [('dma_mask', mask)], 'dma_mask',
                                registers, self._read_stats, timeout_sec)

    def _read_stats(self, reg):
        def r64(lo, hi):
            # Exclusive session: no new job can alter this completed snapshot.
            return (reg.read(hi) << 32) | reg.read(lo)
        return {
            'm00': r64(M00_LO, M00_HI),
            'm10': r64(M10_LO, M10_HI),
            'm01': r64(M01_LO, M01_HI),
            'm20': r64(M20_LO, M20_HI),
            'm02': r64(M02_LO, M02_HI),
            'm11': r64(M11_LO, M11_HI),
            'min_row': reg.read(MIN_ROW),
            'max_row': reg.read(MAX_ROW),
            'min_col': reg.read(MIN_COL),
            'max_col': reg.read(MAX_COL),
            'job_id': reg.read(JOB_ID),
            'status': reg.read(STATUS),
            'ctrl': reg.read(CTRL),
        }
