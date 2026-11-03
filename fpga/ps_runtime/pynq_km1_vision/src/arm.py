"""KM1 host control. No UART access until explicitly enabled; no invented ACK."""
import math
import time

from .arm_timing import validate_timing, move_ms, settle_s


def validate_base_yaw_calibration(calibration):
    """Validate an optional, per-device yaw-to-PWM interpolation table."""
    if not isinstance(calibration, dict) or set(calibration) != {'right_90_pwm', 'center_pwm', 'left_90_pwm'}:
        raise ValueError('Invalid base yaw calibration')
    right, center, left = (calibration[k] for k in ('right_90_pwm', 'center_pwm', 'left_90_pwm'))
    if not all(type(p) is int and 500 <= p <= 2500 for p in (right, center, left)) or not right < center < left:
        raise ValueError('Invalid base yaw calibration')
    return right, center, left


def ik(x, y, z, alpha=None, base_yaw_calibration=None):
    """Vendor V3.1 geometry in mm; keep its empirical base conversion (270/pi)."""
    if not all(math.isfinite(v) for v in (x, y, z)) or y < 0 or (x == 0 and y == 0):
        raise ValueError('Coordinates must be finite, y >= 0, and (x, y) must not be the origin')
    calibration = None if base_yaw_calibration is None else validate_base_yaw_calibration(base_yaw_calibration)
    if y == 0 and calibration is None:
        raise ValueError('Side-axis planning requires base yaw calibration')
    def solve(a):
        r = math.hypot(x, y) - 155 * math.cos(math.radians(a))
        h = z - 100 - 155 * math.sin(math.radians(a))
        d = math.hypot(r, h)
        if h < -100 or d < 1e-8 or not 17 <= d <= 193:
            return None
        b = (d*d + 105**2 - 88**2)/(210*d)
        c = -(d*d - 105**2 - 88**2)/(2*105*88)
        if abs(b) > 1 or abs(c) > 1:
            return None
        shoulder = math.degrees(math.acos(max(-1,min(1,r/d)))) * (-1 if h < 0 else 1)
        shoulder += math.degrees(math.acos(b))
        elbow = 180 - math.degrees(math.acos(c))
        wrist = a - shoulder + elbow
        if not (0 <= shoulder <= 180 and abs(elbow) <= 135 and abs(wrist) <= 90):
            return None
        yaw = math.atan2(x, y)
        if calibration is None:
            base = yaw * 270 / math.pi
            base_pwm = int(1500 - 2000 * base / 270)
        else:
            right, center, left = calibration
            fraction = abs(yaw) / (math.pi / 2)
            endpoint = right if yaw >= 0 else left
            base_pwm = round(center + fraction * (endpoint - center))
        pwm = [base_pwm, int(1500+2000*(shoulder-90)/270),
               int(1500+2000*elbow/270), int(1500+2000*wrist/270)]
        return pwm if all(500 <= p <= 2500 for p in pwm) else None
    angles = [alpha] if alpha is not None else range(-135,1)
    for a in angles:
        if not math.isfinite(a): raise ValueError('Invalid angle')
        out = solve(a)
        if out is not None: return out
    raise ValueError('Unreachable pose: %s' % ((x,y,z,alpha),))


def packet(pairs, duration):
    if not isinstance(duration,int) or not 300 <= duration <= 5000:
        raise ValueError('Move duration outside 300..5000 ms')
    for sid,pwm in pairs:
        if sid not in range(6) or not isinstance(pwm,int) or not 500 <= pwm <= 2500:
            raise ValueError('Invalid servo command')
    return ('{' + ''.join('#%03dP%04dT%04d!' % (i,p,duration) for i,p in pairs) + '}').encode('ascii')


def plan_pick(x,y,label,cfg):
    """Preflight EVERY waypoint before opening UART. Joint-space interpolation is NOT collision checking."""
    a=cfg['arm']; validate_timing(a); gx=a['grab_z']; lift=a['lift_z']
    if label not in a['bins']: raise ValueError('Class has no bin')
    xmin,xmax,ymin,ymax=a['pick_bounds_mm']
    if not (xmin <= x <= xmax and ymin <= y <= ymax): raise ValueError('Outside pick area')
    bx,by,bz=a['bins'][label]; hx,hy,hz=a['home']
    if not (0 <= gx < lift and 0 <= bz < lift): raise ValueError('Invalid heights')
    seq=[]
    def move(name,xx,yy,zz):
        duration=move_ms(a,name)
        pwm=ik(xx,yy,zz,a.get('alpha'),a.get('base_yaw_calibration'))
        seq.append((name,packet(list(enumerate(pwm)),duration),duration))
    def grip(name,p):
        duration=move_ms(a,name)
        seq.append((name,packet([(5,p)],duration),duration))
    move('home',hx,hy,hz)
    duration=move_ms(a,'wrist')
    seq.append(('wrist',packet([(4,a['wrist_pwm'])],duration),duration))
    grip('open',a['open_pwm'])
    move('above_target',x,y,lift); move('descend',x,y,gx)
    grip('close',a['close_pwm']); move('lift',x,y,lift)
    move('above_bin',bx,by,lift); move('lower_bin',bx,by,bz)
    grip('release',a['open_pwm']); move('leave_bin',bx,by,lift)
    move('return',hx,hy,hz)
    return seq


def execute(plan,cfg,enabled=False,before_motion=None):
    if not enabled:
        return [{'step':n,'command':p.decode(),'wait_ms':d} for n,p,d in plan]
    if not cfg['arm'].get('commissioned',False): raise ValueError('Arm not commissioned')
    from .arm_transport import ArmConnection
    from .arm_plan import require_plan
    require_plan(plan,cfg)
    connection=ArmConnection(cfg['arm'])
    # Validate every V3 packet before opening USB, including later waypoints.
    for _,data,duration in plan: connection.validate(data,duration)
    try:
        connection.open()
        if before_motion: before_motion()
        for name,p,d in plan:
            connection.send(p,d,before_motion if name=='above_target' else None)
            time.sleep(d/1000 + settle_s(cfg['arm'], name))
    finally:
        connection.close()
    # Timed sequence completion is not a servo ACK or verified grasp.
    return {'status':'sequence_sent_unverified','steps':len(plan)}
