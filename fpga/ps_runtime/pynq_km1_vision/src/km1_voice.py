"""ASRPRO vocabulary and proposed KM1 UART contract; no PC speech/audio or I/O.

Frames become usable only after reviewed ASRPRO and ESP32 firmware deployment.
Requests select a mode; callers must still run the current execution gates.
"""

COMMANDS = (
    (1, '纸类分拣', 'PAPER', '正在检查纸类分拣条件'),
    (2, '金属分拣', 'METAL', '正在检查金属分拣条件'),
    (3, '塑料分拣', 'PLASTIC', '正在检查塑料分拣条件'),
    (4, '自动分拣', 'AUTO', '正在检查自动分拣条件'),
    (5, '汇报分拣结果', 'REPORT', '正在查询分拣结果'),
    (6, '停止分拣', 'STOP', '已请求停止分拣'),
)

# Only these fixed response tokens may select prerecorded/generated clips.
REPLIES = {
    'PAPER_PASS': '最近纸类任务视觉核验通过',
    'METAL_PASS': '最近金属任务视觉核验通过',
    'PLASTIC_PASS': '最近塑料任务视觉核验通过',
    'REVIEW': '分拣结果需要人工复核',
    'REJECTED': '分拣条件不满足，请检查工位',
    'EMPTY': '取物区未检出目标',
    'NO_RESULT': '暂无可汇报的实物分拣结果',
    'BUSY': '正在处理当前任务，请稍候',
    'STOPPED': '已停止接收新的分拣任务',
    'OFFLINE': '分拣控制连接未就绪',
    'PREVIEW': '已完成预演，尚未执行抓放',
}


def request_frame(token):
    if token not in {c[2] for c in COMMANDS}:
        raise ValueError('unknown_voice_request')
    return ('@KSV1:' + token + '!').encode('ascii')


def parse_request(frame):
    """Require a whole, exact frame; never parse speech or arbitrary serial code."""
    return next((c[2] for c in COMMANDS if frame == request_frame(c[2])), None)


def check_requested_material(token, observed_label):
    """Gate a requested material against fresh vision; never relabel an object."""
    requested = {'PAPER': 'paper', 'METAL': 'metal', 'PLASTIC': 'plastic', 'AUTO': None}
    if token not in requested:
        raise ValueError('request_is_not_a_sorting_mode')
    if observed_label not in ('paper', 'metal', 'plastic'):
        raise ValueError('no_known_material_observed')
    if requested[token] is not None and observed_label != requested[token]:
        raise ValueError('requested_material_not_observed')
    return observed_label


def reply_frame(token):
    if token not in REPLIES:
        raise ValueError('unknown_voice_reply')
    return ('$KSP1:' + token + '!').encode('ascii')


def result_reply(report):
    """Report the supplied latest raw real record, never promote human feedback.

    The caller owns latest-record selection and task binding. Missing/preview/
    simulated records cannot generate a real success announcement.
    """
    if not isinstance(report, dict) or report.get('mode') != 'known_physical':
        return 'NO_RESULT'
    status = report.get('status')
    if status in ('motion_intent_reserved', 'motion_in_progress', 'verifying'):
        return 'BUSY'
    if status == 'needs_review':
        return 'REVIEW'
    if status == 'rejected':
        return 'REJECTED'
    if status == 'empty_scene':
        return 'EMPTY'
    if status == 'completed_visual_evidence':
        choice = report.get('choice')
        label = choice.get('label') if isinstance(choice, dict) else None
        return {'paper': 'PAPER_PASS', 'metal': 'METAL_PASS',
                'plastic': 'PLASTIC_PASS'}.get(label, 'NO_RESULT')
    return 'NO_RESULT'
