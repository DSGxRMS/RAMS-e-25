"""Brake and verify standstill while the EUFS ROS interface is still live."""
import math
import time


TIMEOUT_S = 60.0  # Wall time. A paused simulator cannot confirm a stop.
PUBLISH_INTERVAL_NS = 50_000_000
FEEDBACK_MAX_GAP_NS = 150_000_000
STILL_DWELL_NS = 200_000_000
STILL_SPEED_MPS = 0.05


def brake_to_stop(clock, feedback, publish, ros_ok, actuator_delay_s):
    """Publish braking commands until fresh planar GT confirms standstill."""
    started = time.monotonic()
    deadline = started + TIMEOUT_S
    first_brake = last_brake = previous_stamp = still_since = None
    last_speed = stop_stamp = None
    status = "timeout"
    while ros_ok() and time.monotonic() < deadline:
        now = clock()
        if last_brake is not None and now < last_brake:
            # Do not certify stop across a simulator reset.
            status = "clock_reset"
            break
        if last_brake is None or now - last_brake >= PUBLISH_INTERVAL_NS:
            last_brake = publish()
            if first_brake is None:
                first_brake = last_brake

        sample = feedback()
        if sample is not None:
            stamp, speed = sample
            earliest = first_brake + round(actuator_delay_s * 1e9)
            if (stamp >= earliest and -60_000_000 <= now - stamp <= FEEDBACK_MAX_GAP_NS
                    and math.isfinite(speed) and (previous_stamp is None or stamp > previous_stamp)):
                if previous_stamp is None or stamp - previous_stamp > FEEDBACK_MAX_GAP_NS:
                    still_since = None
                previous_stamp = stamp
                last_speed = speed
                if speed > STILL_SPEED_MPS:
                    still_since = None
                elif still_since is None:
                    still_since = stamp
                elif stamp - still_since >= STILL_DWELL_NS:
                    status, stop_stamp = "confirmed", stamp
                    break
        time.sleep(.01)
    if status == "timeout" and not ros_ok():
        status = "ros_shutdown"
    return dict(status=status, first_brake_sim_ns=first_brake,
                standstill_confirmed_sim_ns=stop_stamp, last_planar_speed_mps=last_speed,
                elapsed_wall_s=time.monotonic() - started)
