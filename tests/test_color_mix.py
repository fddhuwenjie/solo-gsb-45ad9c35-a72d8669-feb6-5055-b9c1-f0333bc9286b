"""端到端测试：POST /color-mix。"""
import math

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def mix(body):
    resp = client.post("/color-mix", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def expect422(body):
    resp = client.post("/color-mix", json=body)
    assert resp.status_code == 422, resp.text
    return resp.json()["detail"]


def expect422_text(body):
    """引擎抛出的 422：detail 为字符串；Pydantic 校验的为数组。"""
    detail = expect422(body)
    assert isinstance(detail, str), detail
    return detail


def rgba(body):
    return mix(body)["rgba"]


# ---------- sRGB 基础混合 ----------


def test_hex_50_50_srgb():
    data = mix({"color_a": "#ff0000", "color_b": "#0000ff"})
    assert data["space"] == "srgb"
    assert data["hue_interpolation"] is None
    # 插值轨迹保留高精度
    assert data["coords"] == [127.5, 0.0, 127.5]
    # 最终 rgba 按 CSS 8-bit 四舍五入：127.5 -> 128
    assert data["rgba"] == {
        "r": 128,
        "g": 0,
        "b": 128,
        "alpha": 1.0,
        "css": "rgb(128 0 128)",
    }


def test_short_hex_and_alpha_hex():
    # 3 位 / 4 位 hex 展开
    data = mix({"color_a": "#f00", "color_b": "#00f8"})
    # color_b alpha = 0x88/255，预乘插值在 sRGB 内
    alpha_b = 0x88 / 255
    a = 0.5 + 0.5 * alpha_b
    r = 0.5 / a * 255
    b = (0.5 * alpha_b / a) * 255
    assert data["rgba"]["alpha"] == pytest.approx(a, abs=1e-6)
    assert data["rgba"]["r"] == math.floor(r + 0.5)
    assert data["rgba"]["b"] == math.floor(b + 0.5)


def test_rgb_syntax_variants_equivalent():
    bodies = [
        {"color_a": "rgb(255, 0, 0)", "color_b": "rgb(0, 0, 255)"},
        {"color_a": "rgba(255 0 0)", "color_b": "rgba(0,0,255)"},
        {"color_a": "rgb(100% 0% 0%)", "color_b": "rgb(0% 0% 100%)"},
    ]
    results = [rgba(b) for b in bodies]
    for got in results:
        assert got["r"] == 128
        assert got["b"] == 128
        assert got["alpha"] == 1.0


def test_rgb_channels_clamped():
    data = mix({"color_a": "rgb(300 -50 0)", "color_b": "rgb(0 0 0)"})
    # 300 裁剪为 255，与 0 混合 50% -> 127.5 -> 128
    assert data["rgba"]["r"] == 128
    assert data["rgba"]["g"] == 0


# ---------- 权重 ----------


def test_weights_default_and_fill():
    data = mix({"color_a": "#f00", "color_b": "#00f"})
    assert data["weights"] == {
        "given": [None, None],
        "resolved": [50.0, 50.0],
        "sum_percent": 100.0,
        "normalized": [0.5, 0.5],
        "alpha_scale": 1.0,
    }
    data = mix({"color_a": "#f00", "color_b": "#00f", "weight_a": 70})
    assert data["weights"]["given"] == [70.0, None]
    assert data["weights"]["resolved"] == [70.0, 30.0]
    assert data["weights"]["normalized"] == [0.7, 0.3]
    # CSS 8-bit 四舍五入 floor(x+0.5)：178.5 -> 179、76.5 -> 77
    assert data["rgba"]["r"] == math.floor(0.7 * 255 + 0.5)
    assert data["rgba"]["b"] == math.floor(0.3 * 255 + 0.5)
    # 百分比字符串与只给 b
    data = mix({"color_a": "#f00", "color_b": "#00f", "weight_b": "70%"})
    assert data["weights"]["resolved"] == [30.0, 70.0]
    assert data["rgba"]["r"] == math.floor(0.3 * 255 + 0.5)
    assert data["rgba"]["b"] == math.floor(0.7 * 255 + 0.5)


def test_weight_sum_below_100_scales_alpha():
    data = mix(
        {"color_a": "#f00", "color_b": "#00f", "weight_a": "30%", "weight_b": 20}
    )
    w = data["weights"]
    assert w["sum_percent"] == 50.0
    assert w["normalized"] == [0.6, 0.4]
    assert w["alpha_scale"] == 0.5
    # 归一化比例混合后再乘 0.5
    assert data["rgba"]["alpha"] == 0.5
    assert data["rgba"]["r"] == 153
    assert data["rgba"]["b"] == 102
    assert data["rgba"]["css"] == "rgb(153 0 102 / 0.5)"


def test_weight_validation_errors():
    assert "负" in expect422_text(
        {"color_a": "#f00", "color_b": "#00f", "weight_a": -1}
    )
    # 只给一个时补全后为负
    assert "负" in expect422_text(
        {"color_a": "#f00", "color_b": "#00f", "weight_a": 120}
    )
    assert "0" in expect422_text(
        {"color_a": "#f00", "color_b": "#00f", "weight_a": 0, "weight_b": 0}
    )
    assert "100" in expect422_text(
        {"color_a": "#f00", "color_b": "#00f", "weight_a": 60, "weight_b": 60}
    )
    # 布尔权重由 Pydantic 在入参层拒绝（detail 为校验错误数组）
    detail = expect422({"color_a": "#f00", "color_b": "#00f", "weight_a": True})
    assert isinstance(detail, list) and "布尔" in str(detail)
    assert "百分比" in expect422_text(
        {"color_a": "#f00", "color_b": "#00f", "weight_a": "abc"}
    )


# ---------- alpha 预乘 ----------


def test_premultiplied_alpha_interpolation():
    data = mix(
        {"color_a": "rgb(255 0 0 / 1)", "color_b": "rgba(0,0,255,0.25)"}
    )
    # 混合 alpha = 0.5*1 + 0.5*0.25 = 0.625
    assert data["rgba"]["alpha"] == pytest.approx(0.625)
    # 预乘 r = 0.5*1/0.625 = 0.8 -> 204；b = 0.5*0.25/0.625 = 0.2 -> 51
    assert data["rgba"]["r"] == pytest.approx(204)
    assert data["rgba"]["b"] == pytest.approx(51)
    pm = data["interpolation"]["premultiplied"]
    assert pm[0] == [255.0, 0.0, 0.0, 1.0]
    assert pm[1][:2] == [0.0, 0.0]
    assert pm[1][2] == pytest.approx(255 * 0.25)
    assert pm[1][3] == 0.25


# ---------- srgb-linear ----------


def test_srgb_linear_midpoint():
    data = mix(
        {"color_a": "#f00", "color_b": "#00f", "interpolation_space": "srgb-linear"}
    )
    # 线性中点再编码：((0.5)**(1/2.4) 路径)
    expected = (1.055 * 0.5 ** (1 / 2.4) - 0.055) * 255
    assert data["coords"][:2] == [0.5, 0.0]
    assert data["rgba"]["r"] == math.floor(expected + 0.5)
    assert data["rgba"]["b"] == math.floor(expected + 0.5)


# ---------- hsl 极坐标空间 ----------


def test_hsl_longer_arc_green_midpoint():
    data = mix(
        {
            "color_a": "hsl(0 100% 50%)",
            "color_b": "hsl(240 100% 50%)",
            "interpolation_space": "hsl",
            "hue_interpolation": "longer",
        }
    )
    assert data["coords"][0] == 120.0  # 长弧经过 360°，中点 120°（绿）
    assert data["rgba"]["g"] == 255
    # 0→240 顺向本身即 240° 长弧，无需修正；但弧线方法已记录
    assert data["hue_interpolation"] == "longer"


def test_hsl_longer_arc_needs_fixup():
    # 蓝(240)→红(0)：顺向差 240° 已是长弧；反向写法红→蓝差 120° 时
    # longer 需把 h2 加 360 绕行，中点落到品红 300°
    data = mix(
        {
            "color_a": "hsl(0 100% 50%)",
            "color_b": "hsl(240 100% 50%)",
            "interpolation_space": "hsl",
            "hue_interpolation": "shorter",
        }
    )
    assert data["coords"][0] == pytest.approx(300.0)
    assert "短弧" in data["interpolation"]["hue_fixup"]

    data = mix(
        {
            "color_a": "hsl(60 100% 50%)",
            "color_b": "hsl(120 100% 50%)",
            "interpolation_space": "hsl",
            "hue_interpolation": "longer",
        }
    )
    # 顺向仅 60°，long 绕行 300°：h2+360=480，中点 270°
    assert data["coords"][0] == pytest.approx(270.0)
    assert "长弧" in data["interpolation"]["hue_fixup"]


def test_hsl_shorter_is_default_arc():
    # 红(0) 与蓝(240)：短弧差 120°，中点 300°（品红方向）
    data = mix(
        {
            "color_a": "hsl(0,100%,50%)",
            "color_b": "hsla(240,100%,50%,1)",
            "interpolation_space": "hsl",
        }
    )
    assert data["hue_interpolation"] == "shorter"
    assert data["coords"][0] == pytest.approx(300.0)
    assert data["rgba"]["r"] == 255
    assert data["rgba"]["b"] == 255


def test_hsl_increasing_decreasing_arcs():
    base = {
        "color_a": "hsl(0,100%,50%)",
        "color_b": "hsl(240,100%,50%)",
        "interpolation_space": "hsl",
    }
    inc = mix({**base, "hue_interpolation": "increasing"})
    assert inc["coords"][0] == pytest.approx(120.0)  # 0 -> 240 递增，中点 120
    dec = mix({**base, "hue_interpolation": "decreasing"})
    assert dec["coords"][0] == pytest.approx(300.0)  # 0 -> -120 递减，折回 300


def test_hsl_achromatic_hue_borrowed():
    data = mix(
        {
            "color_a": "hsl(120 0% 50%)",
            "color_b": "hsl(0 100% 50%)",
            "interpolation_space": "hsl",
        }
    )
    assert "借用 color_b" in data["interpolation"]["missing_hue"]
    # hue 恒为 0，l=50%、s 从 0 增到 50%：hsl(0,50%,50%)
    assert data["coords"] == [0.0, 50.0, 50.0]


def test_hsl_both_achromatic_hue_zero():
    data = mix(
        {
            "color_a": "#ffffff",
            "color_b": "#000000",
            "interpolation_space": "hsl",
        }
    )
    assert "同取 0" in data["interpolation"]["missing_hue"]
    assert data["coords"] == [0.0, 0.0, 50.0]
    assert data["rgba"]["r"] == 128  # 127.5 四舍五入


# ---------- oklab / oklch ----------


def test_oklab_grays_mix_to_neutral():
    data = mix(
        {"color_a": "#fff", "color_b": "#000", "interpolation_space": "oklab"}
    )
    # 白 L=1、黑 L=0 线性中点 L=0.5；sRGB 约 118.4
    assert data["coords"][0] == pytest.approx(0.5, abs=1e-6)
    assert data["rgba"]["r"] == data["rgba"]["g"] == data["rgba"]["b"]


def test_oklab_reference_conversion_coords():
    data = mix(
        {"color_a": "#ff0000", "color_b": "#ff0000", "interpolation_space": "oklab"}
    )
    L, a, b = data["coords"]
    assert L == pytest.approx(0.627955, abs=1e-4)
    assert a == pytest.approx(0.224863, abs=1e-4)
    assert b == pytest.approx(0.125846, abs=1e-4)
    # 两色相同，混合仍是纯红（oklab 往返误差很小）
    assert data["rgba"]["r"] == 255
    assert data["rgba"]["g"] == 0
    assert data["rgba"]["b"] == 0


def test_oklch_achromatic_borrows_hue():
    data = mix(
        {"color_a": "#fff", "color_b": "#f00", "interpolation_space": "oklch"}
    )
    assert "借用 color_b" in data["interpolation"]["missing_hue"]
    # hue 恒为红的 29.23°
    assert data["coords"][2] == pytest.approx(29.2339, abs=1e-3)


def test_oklch_shorter_arc_between_red_and_blue():
    # 红 hue≈29.23、蓝 hue≈264.05；短弧差 234.8>180 需借 360
    data = mix(
        {
            "color_a": "#f00",
            "color_b": "#00f",
            "interpolation_space": "oklch",
            "hue_interpolation": "shorter",
        }
    )
    assert data["interpolation"]["hue_fixup"] is not None
    # 中点约为 (29.23 + (264.05-360))/2 = -33.36 -> 326.64
    assert data["coords"][2] == pytest.approx(326.64, abs=0.5)


def test_oklch_out_of_gamut_clipped():
    # 高彩度插值结果可能超出 sRGB 色域，转回 sRGB 后逐通道裁剪到 [0,255]
    data = mix(
        {
            "color_a": "hsl(300 100% 50%)",
            "color_b": "hsl(180 100% 50%)",
            "interpolation_space": "oklch",
            "hue_interpolation": "shorter",
        }
    )
    for channel in ("r", "g", "b"):
        assert 0.0 <= data["rgba"][channel] <= 255.0


# ---------- 转换与插值轨迹 ----------


def test_conversion_trace_present():
    data = mix(
        {
            "color_a": "hsl(120 50% 25%)",
            "color_b": "#abcdef",
            "interpolation_space": "oklab",
        }
    )
    conv_a, conv_b = data["conversions"]
    assert conv_a["format"] == "hsl"
    assert any("hsl" in step for step in conv_a["steps"])
    assert any("OKLab" in step for step in conv_a["steps"])
    assert conv_b["format"] == "hex"
    assert any("255" in step for step in conv_b["steps"])
    assert len(conv_a["coords"]) == 3
    assert any("OKLab" in step for step in data["back_convert_steps"])


def test_srgb_conversion_trace_minimal():
    data = mix({"color_a": "rgb(1 2 3)", "color_b": "#000"})
    assert data["conversions"][0]["steps"][-1].startswith("插值空间即 sRGB")
    assert data["back_convert_steps"] == ["插值空间即 sRGB，无需转回"]


# ---------- 非法输入 ----------


def test_invalid_colors_422():
    bad = [
        "red",
        "#12",
        "#gg0000",
        "rgb(1 2)",
        "rgb(1, 2)",
        "rgb(1 2 3 4)",  # 现代式 alpha 必须用斜杠
        "rgb(1, 2, 3 / 0.5)",  # 逗号与斜杠混用
        "hsl(10 20 30)",  # s/l 必须百分比
        "xyz(1,2,3)",
        "",
        "   ",
    ]
    for value in bad:
        detail = expect422_text({"color_a": value, "color_b": "#000"})
        assert detail


def test_hsl_legacy_comma_and_angle_units_accepted():
    # 传统逗号式 hsl() 与 deg 单位合法
    data = mix(
        {
            "color_a": "hsl(10deg, 50%, 50%)",
            "color_b": "hsla(10, 50%, 50%, 1)",
            "interpolation_space": "hsl",
        }
    )
    assert data["rgba"]["r"] == 191
    assert data["rgba"]["g"] == 85
    assert data["rgba"]["b"] == 64


def test_hue_interpolation_requires_polar_space():
    detail = expect422(
        {
            "color_a": "#f00",
            "color_b": "#00f",
            "interpolation_space": "srgb",
            "hue_interpolation": "longer",
        }
    )
    assert "极坐标" in detail


def test_bad_space_and_arc_422():
    expect422(
        {
            "color_a": "#f00",
            "color_b": "#00f",
            "interpolation_space": "lab",
        }
    )
    expect422(
        {
            "color_a": "#f00",
            "color_b": "#00f",
            "interpolation_space": "hsl",
            "hue_interpolation": "clockwise",
        }
    )


def test_angle_units_for_hsl_hue():
    # turn / rad / grad 单位
    data = mix(
        {
            "color_a": "hsl(0.5turn 100% 50%)",
            "color_b": "hsl(180deg 100% 50%)",
            "interpolation_space": "hsl",
        }
    )
    assert data["coords"][0] == pytest.approx(180.0)
