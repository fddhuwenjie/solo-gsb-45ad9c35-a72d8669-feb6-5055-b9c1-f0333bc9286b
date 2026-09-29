"""CSS Color 规则的两色混合（color-mix 风格）。

流程遵循 CSS Color 4 §13 插值与 CSS Color 5 color-mix()：

1. 解析输入颜色（hex、rgb()/rgba()、hsl()/hsla()），统一为 sRGB + alpha；
2. 权重补全（两缺各 50、缺一补到 100），给定权重先归一化作为插值比例；
   权重和低于 100% 时以 min(1, 和 / 100) 同比缩小结果 alpha；
3. 两色先转换到目标插值空间（srgb / srgb-linear / hsl / oklab / oklch）；
4. 极坐标空间（hsl / oklch）先按 shorter / longer / increasing /
   decreasing 修正 hue 弧线，无彩色（s=0 或 C=0）一侧的缺失 hue 借用
   另一侧 hue（两侧都缺失则取 0）；
5. 颜色通道（极坐标的 hue 除外）预乘 alpha 后按归一化权重线性插值，
   再用混合 alpha 还原；最后转回 sRGB，逐通道裁剪到 [0, 1]。

非法颜色、负权重或权重和为 0 等抛 ColorMixError（映射 422）。
"""
from __future__ import annotations

import math
import re

from .models import ColorMixRequest, ColorMixResponse

# CSS number 记号（含科学计数法）
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_FUNC_RE = re.compile(r"^(rgb|rgba|hsl|hsla)\(\s*(.*?)\s*\)$", re.IGNORECASE | re.DOTALL)
_NUM_RE = re.compile(rf"^{_NUMBER}$")
_PERCENT_RE = re.compile(rf"^({_NUMBER})%$")
_ANGLE_RE = re.compile(rf"^({_NUMBER})(deg|rad|grad|turn)?$", re.IGNORECASE)

_POLAR = {"hsl", "oklch"}
# 各空间通道是否参与预乘 alpha（极坐标的 hue 不预乘）
_PREMULT_INDEX = {
    "srgb": (0, 1, 2),
    "srgb-linear": (0, 1, 2),
    "oklab": (0, 1, 2),
    "hsl": (1, 2),
    "oklch": (0, 1),
}
_EPS = 1e-7


class ColorMixError(ValueError):
    """请求不满足混合前提（映射为 HTTP 422）。"""


# ------------------------------------------------------------- 数值工具


def _finite(value: float, what: str) -> float:
    if not math.isfinite(value):
        raise ColorMixError(f"{what} 必须为有限数值，收到 {value!r}")
    return value


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _round(value: float) -> float:
    """输出修约：6 位小数并消除 -0.0。"""
    if abs(value) < 5e-9:
        return 0.0
    out = round(value, 6)
    return 0.0 if out == 0 else out


def _fmt(value: float) -> str:
    value = _round(value)
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


# ------------------------------------------------------------- 颜色解析


def _parse_alpha(token: str, what: str) -> float:
    """解析 alpha：裸数字按 0–1、百分比按 0–100%，统一裁剪到 [0, 1]。"""
    token = token.strip()
    match = _NUM_RE.match(token)
    if match is not None:
        value = _finite(float(match.group(0)), what)
        return _clamp(value, 0.0, 1.0)
    match = _PERCENT_RE.match(token)
    if match is not None:
        value = _finite(float(match.group(1)), what)
        return _clamp(value / 100.0, 0.0, 1.0)
    raise ColorMixError(f"{what} {token!r} 不是合法的 alpha（数字或百分比）")


def _split_components(body: str, func: str) -> tuple[list[str], str | None]:
    """拆分函数实参，返回 (3 个颜色分量, alpha 或 None)。

    兼容传统逗号式（rgba(1,2,3,0.5)）与现代空格 + 斜杠式
    （rgb(1 2 3 / 50%)）；两种分隔混用为语法错误。
    """
    if "," in body:
        # 传统逗号式：不允许斜杠 alpha，分量内部不允许空白
        if "/" in body:
            raise ColorMixError(f"{func}() 不能混用逗号分隔与斜杠 alpha")
        parts = [p.strip() for p in body.split(",")]
        if len(parts) not in (3, 4):
            raise ColorMixError(
                f"{func}() 需要 3 或 4 个逗号分隔参数，收到 {len(parts)} 个"
            )
        for part in parts:
            if not part or re.search(r"\s", part):
                raise ColorMixError(f"{func}() 逗号式参数内不允许空白")
        alpha = _parse_alpha(parts[3], f"{func}() 的 alpha") if len(parts) == 4 else None
        return parts[:3], alpha

    # 现代式：斜杠分隔 alpha
    alpha: str | None = None
    if "/" in body:
        segments = body.split("/")
        if len(segments) != 2:
            raise ColorMixError(f"{func}() 只能包含一个 '/' 分隔 alpha")
        body, alpha_token = segments[0].strip(), segments[1].strip()
        if not alpha_token or re.search(r"\s", alpha_token):
            raise ColorMixError(f"{func}() 的斜杠 alpha 必须是单个记号")
        alpha = _parse_alpha(alpha_token, f"{func}() 的 alpha")
    parts = body.split()
    if len(parts) != 3:
        raise ColorMixError(
            f"{func}() 现代式需要 3 个空格分隔分量（斜杠 alpha 可选），"
            f"收到 {len(parts)} 个"
        )
    return parts, alpha


def _parse_rgb(func: str, parts: list[str], alpha: float | None) -> list[float]:
    """解析 rgb 三分量：全部 0–255 数字或全部百分比，统一裁剪到 0–1。"""
    numbers = [_NUM_RE.match(p) for p in parts]
    percents = [_PERCENT_RE.match(p) for p in parts]
    if all(numbers):
        channels = [
            _clamp(_finite(float(m.group(0)), f"{func}() 通道"), 0.0, 255.0) / 255.0
            for m in numbers
        ]
    elif all(percents):
        channels = [
            _clamp(_finite(float(m.group(1)), f"{func}() 通道百分比"), 0.0, 100.0)
            / 100.0
            for m in percents
        ]
    else:
        raise ColorMixError(
            f"{func}() 的三个通道必须全部为 0–255 数字或全部为百分比，"
            "不能混用或带其他单位"
        )
    channels.append(1.0 if alpha is None else alpha)
    return channels


def _parse_hue(token: str, what: str) -> float:
    """解析色相角：裸数字按度，支持 deg / rad / grad / turn，结果为度。"""
    match = _ANGLE_RE.match(token.strip())
    if match is None:
        raise ColorMixError(
            f"{what} {token!r} 不是合法色相角（数字或数字 + deg/rad/grad/turn）"
        )
    value = _finite(float(match.group(1)), what)
    unit = (match.group(2) or "deg").lower()
    if unit == "deg":
        return value
    if unit == "rad":
        return value * 180.0 / math.pi
    if unit == "grad":
        return value * 360.0 / 400.0
    return value * 360.0  # turn


def _parse_sl(token: str, what: str) -> float:
    """解析饱和度 / 亮度：必须为百分比，裁剪到 [0, 1]（分数）。"""
    match = _PERCENT_RE.match(token.strip())
    if match is None:
        raise ColorMixError(f"{what} {token!r} 必须是百分比（如 50%）")
    return _clamp(_finite(float(match.group(1)), what), 0.0, 100.0) / 100.0


def _hsl_to_srgb(h: float, s: float, light: float, alpha: float | None) -> list[float]:
    """HSL（h 为度，s/l 为 0–1 分数）转 sRGB，返回 [r, g, b, a]。"""

    def hue_to_rgb(p: float, q: float, t: float) -> float:
        if t < 0.0:
            t += 1.0
        if t > 1.0:
            t -= 1.0
        if t < 1.0 / 6.0:
            return p + (q - p) * 6.0 * t
        if t < 0.5:
            return q
        if t < 2.0 / 3.0:
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0
        return p

    if s == 0.0:
        rgb = [light, light, light]
    else:
        q = light * (1.0 + s) if light >= 0.5 else light + s - light * s
        p = 2.0 * light - q
        hk = (h % 360.0) / 360.0
        rgb = [
            hue_to_rgb(p, q, hk + 1.0 / 3.0),
            hue_to_rgb(p, q, hk),
            hue_to_rgb(p, q, hk - 1.0 / 3.0),
        ]
    rgb.append(1.0 if alpha is None else alpha)
    return rgb


def parse_color(text: object) -> dict:
    """把输入字符串解析为统一结构：

    {format: hex/rgb/hsl, input, srgba: [r, g, b, a]（sRGB 0–1）}
    """
    if not isinstance(text, str) or not text.strip():
        raise ColorMixError("颜色必须是非空字符串")
    raw = text.strip()

    hex_match = _HEX_RE.match(raw)
    if hex_match is not None:
        digits = hex_match.group(1)
        if len(digits) in (3, 4):
            values = [int(ch * 2, 16) for ch in digits]
        else:
            values = [int(digits[i : i + 2], 16) for i in range(0, len(digits), 2)]
        channels = [v / 255.0 for v in values[:3]]
        alpha = values[3] / 255.0 if len(values) == 4 else 1.0
        return {"format": "hex", "input": raw, "srgba": channels + [alpha]}

    func_match = _FUNC_RE.match(raw)
    if func_match is None:
        raise ColorMixError(
            f"颜色 {raw!r} 无法识别：仅支持 hex 与 rgb()/rgba()/hsl()/hsla()"
        )
    func = func_match.group(1).lower()
    parts, alpha = _split_components(func_match.group(2), func)

    if func in ("rgb", "rgba"):
        return {"format": "rgb", "input": raw, "srgba": _parse_rgb(func, parts, alpha)}

    hue = _parse_hue(parts[0], "hsl() 的色相")
    sat = _parse_sl(parts[1], "hsl() 的饱和度")
    light = _parse_sl(parts[2], "hsl() 的亮度")
    return {
        "format": "hsl",
        "input": raw,
        "srgba": _hsl_to_srgb(hue, sat, light, alpha),
    }


# ------------------------------------------------------------- 空间转换


def srgb_to_linear(c: float) -> float:
    """sRGB 去伽马（EOTF^-1），0–1。"""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    """线性光 sRGB 伽马编码，0–1。"""
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055


def linear_to_xyz(r: float, g: float, b: float) -> tuple[float, float, float]:
    """线性 sRGB（D65）转 CIE XYZ（CSS Color 4 sRGB-to-XYZD65 矩阵）。"""
    x = 0.41239079926595934 * r + 0.357584339383878 * g + 0.1804807884018343 * b
    y = 0.21263900587151027 * r + 0.715168678767756 * g + 0.07219231536073371 * b
    z = 0.01933081871559182 * r + 0.11919477979462598 * g + 0.9505321522496607 * b
    return x, y, z


def xyz_to_linear(x: float, y: float, z: float) -> tuple[float, float, float]:
    """CIE XYZ（D65）转线性 sRGB（CSS Color 4 XYZD65-to-linear-sRGB 矩阵）。"""
    r = 3.2409699419045226 * x - 1.5373831775700939 * y - 0.4986107602930034 * z
    g = -0.9692436362808796 * x + 1.8759675015077202 * y + 0.04155505740717559 * z
    b = 0.05563007969699366 * x - 0.20397695888897652 * y + 1.0569715142428786 * z
    return r, g, b


def xyz_to_oklab(x: float, y: float, z: float) -> tuple[float, float, float]:
    """XYZ 经 LMS 非线性压缩转 OKLab（矩阵已按 D65 白点归一化）。"""
    l = 0.8190224379967030 * x + 0.3619062600528904 * y - 0.1288737815209279 * z
    m = 0.0329836539323885 * x + 0.9292868615863434 * y + 0.0361446663506424 * z
    s = 0.0481771893596242 * x + 0.2642395317527308 * y + 0.6335478284694309 * z
    lp, mp, sp = math.cbrt(l), math.cbrt(m), math.cbrt(s)
    L = 0.2104542683093140 * lp + 0.7936177747023054 * mp - 0.0040720430116194 * sp
    a = 1.9779985324311684 * lp - 2.4285922420422590 * mp + 0.4505937096110906 * sp
    b = 0.0259040424655478 * lp + 0.7827717124575296 * mp - 0.8086757549230774 * sp
    return L, a, b


def oklab_to_xyz(L: float, a: float, b: float) -> tuple[float, float, float]:
    """OKLab 反变换回 XYZ（D65），系数与正向矩阵精确配对。"""
    lp = L + 0.3963377773762153 * a + 0.2158037573068383 * b
    mp = L - 0.1055613458156693 * a - 0.0638541728249944 * b
    sp = L - 0.0894841775298209 * a - 1.2914855480187153 * b
    l, m, s = lp ** 3, mp ** 3, sp ** 3
    x = 1.2268798758459298 * l - 0.5578149944601861 * m + 0.2813910456658479 * s
    y = -0.0405757452148011 * l + 1.1122868032803161 * m - 0.0717110580655126 * s
    z = -0.0763729366746604 * l - 0.4214933324022452 * m + 1.5869240198367891 * s
    return x, y, z


def oklab_to_oklch(L: float, a: float, b: float) -> tuple[float, float, float]:
    c = math.hypot(a, b)
    h = 0.0 if c < _EPS else (math.degrees(math.atan2(b, a)) % 360.0)
    return L, c, h


def oklch_to_oklab(L: float, c: float, h: float) -> tuple[float, float, float]:
    if c < _EPS:
        return L, 0.0, 0.0
    rad = math.radians(h % 360.0)
    return L, c * math.cos(rad), c * math.sin(rad)


def srgb_to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    """sRGB（0–1）转 HSL：h 为度，s/l 为 0–1 分数。"""
    hi, lo = max(r, g, b), min(r, g, b)
    d = hi - lo
    light = (hi + lo) / 2.0
    if d < _EPS:
        return 0.0, 0.0, light
    sat = d / (2.0 - hi - lo) if light > 0.5 else d / (hi + lo)
    if hi == r:
        offset = (g - b) / d + (6.0 if g < b else 0.0)
    elif hi == g:
        offset = (b - r) / d + 2.0
    else:
        offset = (r - g) / d + 4.0
    return offset * 60.0, sat, light


def to_space(srgba: list[float], space: str) -> list[float]:
    """统一的 sRGB + alpha 表示转目标空间坐标（含 alpha）。

    hsl 为 [h 度, s 0–1, l 0–1]；srgb / srgb-linear 通道为 0–1。
    """
    r, g, b, a = srgba
    if space == "srgb":
        return [r, g, b, a]
    if space == "srgb-linear":
        return [srgb_to_linear(c) for c in (r, g, b)] + [a]
    lin = [srgb_to_linear(c) for c in (r, g, b)]
    if space == "hsl":
        h, s, light = srgb_to_hsl(r, g, b)
        return [h, s, light, a]
    x, y, z = linear_to_xyz(*lin)
    L, oa, ob = xyz_to_oklab(x, y, z)
    if space == "oklab":
        return [L, oa, ob, a]
    return list(oklab_to_oklch(L, oa, ob)) + [a]


def from_space(coords: list[float], space: str) -> list[float]:
    """目标空间坐标（含 alpha）转回 sRGB + alpha（未裁剪）。"""
    *channels, a = coords
    if space == "srgb":
        return channels + [a]
    if space == "srgb-linear":
        return [linear_to_srgb(c) for c in channels] + [a]
    if space == "hsl":
        return _hsl_to_srgb(channels[0], channels[1], channels[2], a)
    if space == "oklch":
        L, oa, ob = oklch_to_oklab(*channels)
    else:
        L, oa, ob = channels
    x, y, z = oklab_to_xyz(L, oa, ob)
    lin = xyz_to_linear(x, y, z)
    return [linear_to_srgb(c) for c in lin] + [a]


def display_coords(coords: list[float], space: str) -> list[float]:
    """内部工作坐标转展示单位：sRGB 0–255、HSL 的 s/l 为百分比数值。"""
    out = list(coords)
    if space == "srgb":
        out = [c * 255.0 for c in out]
    elif space == "hsl":
        out = [out[0], out[1] * 100.0, out[2] * 100.0]
    return [_round(c) for c in out]


# ------------------------------------------------------------- 转换轨迹


def _convert_steps(src_format: str, space: str) -> list[str]:
    """由输入格式到目标空间的逐步说明（sRGB 为统一中转）。"""
    steps: list[str] = []
    if src_format == "hsl":
        steps.append("hsl() 按色相/饱和度/亮度公式先转换为 sRGB")
    elif src_format == "hex":
        steps.append("hex 的 8-bit 通道归一化为 sRGB（各通道除以 255）")
    else:
        steps.append("rgb() 分量归一化为 sRGB（0–255 / 百分比统一为 0–1）")
    if space == "srgb":
        steps.append("插值空间即 sRGB，无需进一步转换")
    elif space == "srgb-linear":
        steps.append("逐通道应用 sRGB 传递函数逆（去伽马）转为线性光 sRGB")
    elif space == "hsl":
        steps.append("由 sRGB 的 max/min 分量求 HSL（h 单位为度，s/l 取 0–1）")
    else:
        steps += [
            "sRGB 去伽马转为线性光 sRGB",
            "线性 sRGB 经 D65 矩阵转换为 CIE XYZ",
            "XYZ 经 LMS 非线性压缩转换为 OKLab",
        ]
        if space == "oklch":
            steps.append(
                "OKLab 直角坐标极坐标化为 OKLCh（C=hypot(a,b)，h=atan2(b,a)）"
            )
    return steps


def _back_steps(space: str) -> list[str]:
    if space == "srgb":
        return ["插值空间即 sRGB，无需转回"]
    if space == "srgb-linear":
        return ["线性光 sRGB 重新应用 sRGB 传递函数（伽马编码）"]
    if space == "hsl":
        return ["HSL 按色相/饱和度/亮度公式转回 sRGB"]
    steps: list[str] = []
    if space == "oklch":
        steps.append("OKLCh 反极坐标化为 OKLab（a=C·cos h，b=C·sin h）")
    steps += [
        "OKLab 经 LMS 立方与 D65 矩阵转换回 CIE XYZ",
        "XYZ 经矩阵转换回线性光 sRGB",
        "线性光 sRGB 伽马编码为 sRGB",
    ]
    return steps


# ------------------------------------------------------------- 权重


def _parse_weight(value: object, what: str) -> float:
    if value is None:
        return None  # type: ignore[return-value]
    if isinstance(value, bool):
        raise ColorMixError(f"{what} 不能是布尔值")
    if isinstance(value, (int, float)):
        return _finite(float(value), what)
    if isinstance(value, str):
        text = value.strip()
        match = _NUM_RE.match(text) or _PERCENT_RE.match(text)
        if match is None:
            raise ColorMixError(f"{what} {value!r} 不是合法百分比（数字或 '数字%'）")
        token = match.group(1) if match.lastindex else match.group(0)
        return _finite(float(token), what)
    raise ColorMixError(f"{what} 必须是数字或百分比字符串")


def normalize_weights(wa_raw: object, wb_raw: object) -> dict:
    """权重补全、校验与归一化。"""
    wa = _parse_weight(wa_raw, "weight_a")
    wb = _parse_weight(wb_raw, "weight_b")
    given = [wa, wb]
    if wa is None and wb is None:
        resolved = [50.0, 50.0]
    elif wa is None:
        resolved = [100.0 - wb, wb]
    elif wb is None:
        resolved = [wa, 100.0 - wa]
    else:
        resolved = [wa, wb]

    for i, w in enumerate(resolved):
        if w < 0:
            raise ColorMixError(
                f"权重不能为负值（color_{'a' if i == 0 else 'b'} 补全后为 {_r6(w)}%）"
            )
    total = resolved[0] + resolved[1]
    if total <= _EPS:
        raise ColorMixError("两个权重之和为 0，无法混合")
    if total > 100.0 + 1e-9:
        raise ColorMixError(
            f"给定权重之和为 {_r6(total)}%，超过 100%；color-mix() 权重和不得超过 100%"
        )
    normalized = [resolved[0] / total, resolved[1] / total]
    return {
        "given": [None if g is None else _r6(g) for g in given],
        "resolved": [_r6(w) for w in resolved],
        "sum_percent": _r6(total),
        "normalized": [_r6(p) for p in normalized],
        "alpha_scale": _r6(min(1.0, total / 100.0)),
        "_t": normalized[1],
        "_alpha_scale": min(1.0, total / 100.0),
    }


def _r6(value: float) -> float:
    return round(value, 6)


# ------------------------------------------------------------- hue 弧线


def fixup_hue(
    h1: float, h2: float, missing1: bool, missing2: bool, method: str
) -> tuple[float, float, str | None, str | None]:
    """按 CSS Color 4 规则处理缺失 hue 与 hue 弧线。

    返回修正后的 (h1, h2, 缺失借用说明, 弧线修正说明)。h2 可能落在
    [0, 360) 之外以表达绕行方向，插值后再统一折回。
    """
    missing_note: str | None = None
    if missing1 and missing2:
        h1, h2 = 0.0, 0.0
        missing_note = "两侧在目标空间均无彩色（缺失 hue），hue 同取 0"
    elif missing1:
        h1 = h2
        missing_note = "color_a 无彩色（缺失 hue），借用 color_b 的 hue"
    elif missing2:
        h2 = h1
        missing_note = "color_b 无彩色（缺失 hue），借用 color_a 的 hue"

    fix_note: str | None = None
    delta = h1 - h2
    if method == "shorter":
        # 短弧：把 |h1-h2| 收敛到 <=180；恰为 180 时按规范保持不变
        if delta > 180.0:
            h2 += 360.0
            fix_note = "shorter：hue 差大于 180°，color_b 的 hue 加 360° 取短弧"
        elif delta < -180.0:
            h2 -= 360.0
            fix_note = "shorter：hue 差小于 -180°，color_b 的 hue 减 360° 取短弧"
    elif method == "longer":
        # 长弧：把 |h1-h2| 展开到 >=180；恰为 180 时长弧短弧重合，保持不变
        if 0.0 <= delta < 180.0:
            h2 += 360.0
            fix_note = "longer：hue 差在 [0°, 180°)，color_b 的 hue 加 360° 取长弧"
        elif -180.0 < delta < 0.0:
            h2 -= 360.0
            fix_note = "longer：hue 差在 (-180°, 0°)，color_b 的 hue 减 360° 取长弧"
    elif method == "increasing":
        if h2 < h1:
            h2 += 360.0
            fix_note = "increasing：要求 hue 单调增大，color_b 的 hue 加 360°"
    else:  # decreasing
        if h2 > h1:
            h2 -= 360.0
            fix_note = "decreasing：要求 hue 单调减小，color_b 的 hue 减 360°"
    return h1, h2, missing_note, fix_note


# ------------------------------------------------------------- 总入口


def mix_colors(req: ColorMixRequest) -> ColorMixResponse:
    space = req.interpolation_space
    method = req.hue_interpolation
    if method is not None and space not in _POLAR:
        raise ColorMixError(
            f"hue_interpolation 仅可用于极坐标插值空间 hsl / oklch，"
            f"当前空间为 {space}"
        )
    if method is None and space in _POLAR:
        method = "shorter"

    weights = normalize_weights(req.weight_a, req.weight_b)
    t = weights.pop("_t")
    alpha_scale = weights.pop("_alpha_scale")

    parsed = [parse_color(req.color_a), parse_color(req.color_b)]
    conversions = []
    coords = []  # 目标空间内部坐标（含 alpha）
    for color in parsed:
        internal = to_space(color["srgba"], space)
        coords.append(internal)
        conversions.append(
            {
                "input": color["input"],
                "format": color["format"],
                "alpha": _round(color["srgba"][3]),
                "steps": _convert_steps(color["format"], space),
                "coords": display_coords(internal[:-1], space),
            }
        )

    c1, c2 = coords[0], coords[1]
    a1, a2 = c1[-1], c2[-1]

    # 极坐标空间：缺失 hue 借用 + 弧线修正
    missing_note = fix_note = None
    if space in _POLAR:
        # hsl 以 s（通道 1）、oklch 以 c（通道 1）判定无彩色
        h1, h2, missing_note, fix_note = fixup_hue(
            c1[0 if space == "hsl" else 2],
            c2[0 if space == "hsl" else 2],
            c1[1] < _EPS,
            c2[1] < _EPS,
            method,
        )
        c1[0 if space == "hsl" else 2] = h1
        c2[0 if space == "hsl" else 2] = h2

    # 预乘 alpha（极坐标 hue 不参与预乘）
    pm_index = _PREMULT_INDEX[space]
    pm1, pm2 = list(c1[:-1]), list(c2[:-1])
    for i in pm_index:
        pm1[i] *= a1
        pm2[i] *= a2

    mixed_alpha = a1 * (1.0 - t) + a2 * t
    mixed: list[float] = []
    for i in range(3):
        if i in pm_index:
            pm_mix = pm1[i] * (1.0 - t) + pm2[i] * t
            if mixed_alpha > _EPS:
                mixed.append(pm_mix / mixed_alpha)
            else:
                # 混合 alpha 为 0：还原无定义，退化为未预乘线性插值
                mixed.append(c1[i] * (1.0 - t) + c2[i] * t)
        else:
            mixed.append(c1[i] * (1.0 - t) + c2[i] * t)

    # 极坐标：弧线修正可能使 hue 离开 [0, 360)，折回后再转回
    if space in _POLAR:
        mixed[0 if space == "hsl" else 2] %= 360.0

    interpolation = {
        "space": space,
        "premultiplied": [
            display_coords(pm1, space) + [_round(a1)],
            display_coords(pm2, space) + [_round(a2)],
        ],
        "hue_fixup": fix_note,
        "missing_hue": missing_note,
        "mixed_coords": display_coords(mixed, space),
    }

    back_steps = _back_steps(space)
    result = from_space(mixed + [mixed_alpha], space)
    # 转回 sRGB 后逐通道裁剪；alpha 还要乘权重和系数
    result_rgb = [_clamp(c, 0.0, 1.0) for c in result[:3]]
    result_alpha = _clamp(mixed_alpha * alpha_scale, 0.0, 1.0)

    # 最终 rgba 按 CSS rgb() 序列化规则：8-bit 通道四舍五入为整数
    r255, g255, b255 = (int(math.floor(c * 255.0 + 0.5)) for c in result_rgb)
    a_out = _round(result_alpha)
    if a_out == 1.0:
        css = f"rgb({r255} {g255} {b255})"
    else:
        css = f"rgb({r255} {g255} {b255} / {_fmt(a_out)})"

    return ColorMixResponse(
        space=space,
        hue_interpolation=method,
        weights=weights,
        conversions=conversions,
        interpolation=interpolation,
        back_convert_steps=back_steps,
        coords=display_coords(mixed, space),
        rgba={"r": r255, "g": g255, "b": b255, "alpha": a_out, "css": css},
    )
