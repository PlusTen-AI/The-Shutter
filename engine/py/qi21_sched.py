# Sampling schedules as ComfyUI computes them (fp32), stdlib only.
import math
import struct


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def flux_simple(steps, shift=0.69, table=10000):
    """ModelSamplingFlux(shift) + "simple" scheduler: every (table/steps)-th entry from the top, then 0."""
    e = f32(math.exp(shift))
    def sigma(i):  # ts = arange(1, table+1) / table in fp32; e / (e + (1/t - 1)) in fp32
        t = f32(f32(i) / f32(table))
        # torch evaluates scalar / tensor as reciprocal(tensor) * scalar
        return f32(f32(1.0 / f32(e + f32(f32(1.0 / t) - 1.0))) * e)
    ss = table / steps
    return [sigma(table - int(x * ss)) for x in range(steps)] + [0.0]


def manual(text):
    """ManualSigmas node: numbers in a string."""
    import re
    return [f32(float(v)) for v in re.findall(r"[-+]?(?:\d*\.*\d+)", text)]


TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"
