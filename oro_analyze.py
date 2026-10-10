# -*- coding: utf-8 -*-
"""
油红O (Oil Red O) 切片脂滴定量分析
---------------------------------
输出口径:
  ratio_image  = 脂滴面积 / 整张图面积        (本次任务指定口径)
  ratio_tissue = 脂滴面积 / 组织面积          (参考对照, 排除白色背景)

用法:
  python oro_analyze.py --input <img_or_dir> --out <out_dir> [--mode auto|mix|a|red]
                        [--thr 0.0-1.0 手动阈值] [--min-area 20] [--watershed] [--group-from-name]

依赖: numpy opencv-python-headless scikit-image matplotlib pandas
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 中文字体 ----
for _f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ---------------------------------------------------------------- 工具函数
def imread_unicode(path: Path) -> np.ndarray:
    """cv2.imread 不支持中文路径, 用 fromfile 绕过."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法解码图像: {path}")
    return img


def otsu_threshold(scores: np.ndarray) -> float:
    """在 0-255 uint8 单通道上做 Otsu, 返回阈值."""
    u8 = np.clip(scores, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr)


def red_dominance_score(rgb: np.ndarray) -> np.ndarray:
    """红度打分: R - max(G,B), 油红O红色染料在此通道上最亮."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    return r - np.maximum(g, b)


def saturation_score(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 1].astype(np.float32)


def a_channel(rgb: np.ndarray) -> np.ndarray:
    """Lab 的 a* 通道: >128 偏红."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[:, :, 1].astype(np.float32)


def background_mask(rgb: np.ndarray) -> np.ndarray:
    """空白玻片背景: 既亮又几乎无饱和度."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    return (v > 225) & (s < 26)


# ---------------------------------------------------------------- 分割核心
def segment(img_bgr: np.ndarray, mode: str = "fixed", manual_thr: float | None = None,
            min_area: int = 20, use_watershed: bool = False, tighten: int = 0):
    """返回 (mask_bool, debug_dict). mask_bool 为 True = 脂滴.

    mode="fixed": 红度 R-max(G,B) > 绝对阈值 (跨图像统一, 组间可比, 推荐).
                  油红O染料 red≈+63~+107, 苏木素紫质 ≈-15~-74, 淡粉胞质 ≈+16,
                  阈值 40 可干净分离. manual_thr 缺省时用 40.
    其余为 Otsu 自适应模式, 仅用于单图调参, 不建议组间比较.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    a = a_channel(rgb)
    red = red_dominance_score(rgb)
    sat = saturation_score(rgb)

    thr_a = otsu_threshold(a)
    thr_red = otsu_threshold(np.clip(red, 0, 255))
    thr_sat = otsu_threshold(sat)

    if mode == "fixed":
        eff_thr = manual_thr if manual_thr is not None else 40.0
        base = red > eff_thr
        score = red
    elif mode == "a":
        score, base = a, a > thr_a
    elif mode == "red":
        score, base = red, red > thr_red
    elif mode == "sat":
        score, base = sat, sat > thr_sat
    else:  # mix: 多数投票
        votes = (a > thr_a).astype(np.uint8) + (red > thr_red).astype(np.uint8) + (sat > thr_sat).astype(np.uint8)
        base = votes >= 2
        score = a

    if manual_thr is not None and mode != "fixed":
        base = score > manual_thr

    mask = base.astype(np.uint8)

    # 形态学清理: 闭运算补内部空洞 -> 开运算去毛刺
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # 边缘收敛: 抵消成像模糊/抗锯齿造成的红色外扩 (合成真值校准, 默认关闭)
    if tighten > 0:
        mask = cv2.erode(mask, k, iterations=tighten)

    # 去碎点
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    mask = keep

    if use_watershed and mask.any():
        mask = _watershed_split(rgb, mask)

    dbg = {
        "a": a, "red": red, "sat": sat, "score": score,
        "thr_a": thr_a, "thr_red": thr_red, "thr_sat": thr_sat,
        "eff_thr": (manual_thr if (mode == "fixed" and manual_thr is not None) else
                    40.0 if mode == "fixed" else
                    {"a": thr_a, "red": thr_red, "sat": thr_sat, "mix": thr_a}[mode]),
        "mode": mode,
        "bg": background_mask(rgb),
    }
    return mask.astype(bool), dbg


def _watershed_split(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """用距离变换 + 分水岭把粘连脂滴拆开."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    dist = ndi.distance_transform_edt(mask)
    coords = peak_local_max(dist, labels=mask, min_distance=3, exclude_border=False)
    markers = np.zeros(dist.shape, dtype=np.int32)
    for idx, (rr, cc) in enumerate(coords, start=1):
        markers[rr, cc] = idx
    if markers.max() == 0:
        return mask
    labels = watershed(-dist, markers, mask=mask)
    return labels > 0


# ---------------------------------------------------------------- 指标
@dataclass
class Metrics:
    name: str
    group: str = "all"
    width: int = 0
    height: int = 0
    img_area: int = 0
    tissue_area: int = 0
    droplet_area: int = 0
    droplet_count: int = 0
    mean_droplet_area: float = 0.0
    median_droplet_area: float = 0.0
    max_droplet_area: int = 0
    ratio_image: float = 0.0
    ratio_tissue: float = 0.0
    thr_a: float = 0.0
    thr_red: float = 0.0
    thr_sat: float = 0.0

    def as_row(self) -> dict:
        return {
            "图像": self.name, "分组": self.group,
            "宽": self.width, "高": self.height,
            "全图面积_px": self.img_area,
            "组织面积_px": self.tissue_area,
            "脂滴面积_px": self.droplet_area,
            "脂滴个数": self.droplet_count,
            "平均脂滴面积_px": round(self.mean_droplet_area, 2),
            "中位脂滴面积_px": round(self.median_droplet_area, 2),
            "最大脂滴面积_px": self.max_droplet_area,
            "脂滴面积占比_全图_%": round(self.ratio_image * 100, 3),
            "脂滴面积占比_组织_%": round(self.ratio_tissue * 100, 3),
        }


def parse_group(name: str) -> str:
    """从文件名推断分组: 去掉勾选标记/尾部 (1)/(2)/序号/扩展名.
    如 'SG3 √.tif'->SG, 'SGA+3O 1.tif'->SGA+3O, 'PA+3O  (1).tif'->PA+3O"""
    stem = Path(name).stem
    stem = stem.replace("√", "").replace("✓", "").strip()
    stem = re.sub(r"\s*\(\s*\d+\s*\)\s*$", "", stem)   # 尾部 (1)
    stem = re.sub(r"[\s_\-]*\d+$", "", stem).strip()   # 尾部纯数字
    return stem or "all"


def analyze_one(path: Path, mode: str, manual_thr, min_area: int,
                use_watershed: bool, group_from_name: bool,
                tighten: int = 0) -> tuple[Metrics, dict, np.ndarray, np.ndarray]:
    img_bgr = imread_unicode(path)
    mask, dbg = segment(img_bgr, mode, manual_thr, min_area, use_watershed, tighten)
    h, w = img_bgr.shape[:2]
    img_area = h * w
    droplet_area = int(mask.sum())
    tissue = dbg["bg"] == False  # noqa: E712
    tissue_area = int(tissue.sum())
    if tissue_area == 0:
        tissue_area = img_area

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)] if n > 1 else []

    m = Metrics(
        name=path.name,
        group=parse_group(path.name) if group_from_name else "all",
        width=w, height=h,
        img_area=img_area,
        tissue_area=tissue_area,
        droplet_area=droplet_area,
        droplet_count=len(areas),
        mean_droplet_area=float(np.mean(areas)) if areas else 0.0,
        median_droplet_area=float(np.median(areas)) if areas else 0.0,
        max_droplet_area=int(max(areas)) if areas else 0,
        ratio_image=droplet_area / img_area,
        ratio_tissue=droplet_area / tissue_area,
        thr_a=dbg["thr_a"], thr_red=dbg["thr_red"], thr_sat=dbg["thr_sat"],
    )
    return m, dbg, mask, img_bgr


# ---------------------------------------------------------------- 出图
def make_panel(img_bgr, mask, dbg, m: Metrics, out_png: Path):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    overlay[mask] = (overlay[mask] * 0.35 + np.array([255, 255, 0]) * 0.65).astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    chan_name = {"fixed": "红度 R-max(G,B)", "red": "红度 R-max(G,B)",
                 "a": "Lab a* 通道", "sat": "HSV 饱和度", "mix": "Lab a* 通道"}[dbg["mode"]]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6), dpi=130)
    axes[0].imshow(rgb); axes[0].set_title(f"原图  {m.width}×{m.height}")
    axes[1].imshow(dbg["score"], cmap="RdYlBu_r"); axes[1].set_title(
        f"{chan_name}  (阈值={dbg['eff_thr']:.0f})")
    axes[2].imshow(mask, cmap="gray"); axes[2].set_title(
        f"脂滴掩膜  占比(全图)={m.ratio_image*100:.2f}%")
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); axes[3].set_title(
        f"叠加  个数={m.droplet_count}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(m.name, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_summary(df: pd.DataFrame, out_png: Path):
    if df.empty:
        return
    d = df.copy()
    d["占比"] = d["脂滴面积占比_全图_%"]
    groups = list(dict.fromkeys(d["分组"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=130)

    # 1 每张图柱状
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
    cmap = {g: colors[i % len(colors)] for i, g in enumerate(groups)}
    ax.bar(range(len(d)), d["占比"], color=[cmap[g] for g in d["分组"]])
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["图像"], rotation=90, fontsize=7)
    ax.set_ylabel("脂滴面积占比 (%)")
    ax.set_title("各图像脂滴面积占比 (分母=整张图)")
    ax.axhline(d["占比"].mean(), ls="--", lw=1, color="#888")
    ax.text(0.99, d["占比"].mean(), f" 均值 {d['占比'].mean():.2f}%",
            transform=ax.get_yaxis_transform(), va="bottom", ha="right", fontsize=8)

    # 2 按分组箱线
    ax = axes[1]
    data = [d.loc[d["分组"] == g, "占比"].values for g in groups]
    try:  # matplotlib >= 3.9 用 tick_labels, 旧版用 labels
        bp = ax.boxplot(data, patch_artist=True, tick_labels=groups, widths=0.55)
    except TypeError:
        bp = ax.boxplot(data, patch_artist=True, labels=groups, widths=0.55)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(cmap[g]); patch.set_alpha(0.75)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-.09, .09, len(vals)),
                   vals, s=18, color="#222", zorder=3)
    ax.set_ylabel("脂滴面积占比 (%)")
    ax.set_title("分组分布")
    ax.tick_params(axis="x", labelsize=8)

    # 3 脂滴个数
    ax = axes[2]
    ax.bar(range(len(d)), d["脂滴个数"], color=[cmap[g] for g in d["分组"]])
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["图像"], rotation=90, fontsize=7)
    ax.set_ylabel("脂滴个数")
    ax.set_title("检出脂滴颗粒数 (min_area 过滤后)")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_report_html(df: pd.DataFrame, panels: list[Path], out_html: Path, params: dict):
    rows = "".join(
        "<tr>" + "".join(f"<td>{r[c]}</td>" for c in df.columns) + "</tr>"
        for _, r in df.iterrows()
    )
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    imgs = "".join(
        f'<figure><img src="{p.name}" alt="{p.name}"><figcaption>{p.stem}</figcaption></figure>'
        for p in panels
    )
    pstr = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in params.items())
    mean_all = df["脂滴面积占比_全图_%"].mean() if len(df) else 0
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>油红O 脂滴面积占比分析报告</title>
<style>
 body{{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:0;padding:32px;
      background:#0f1115;color:#e8eaed}}
 h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:32px 0 10px;color:#9ab8ff}}
 .sub{{color:#9aa0a6;font-size:13px;margin-bottom:24px}}
 .kpi{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
 .card{{background:#1a1d23;border:1px solid #2a2f37;border-radius:10px;padding:14px 20px;min-width:150px}}
 .card .v{{font-size:26px;font-weight:700;color:#7ee787}}
 .card .l{{font-size:12px;color:#9aa0a6;margin-top:2px}}
 .params{{background:#1a1d23;border:1px solid #2a2f37;border-radius:10px;padding:14px 22px;font-size:13px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
 th,td{{border:1px solid #2a2f37;padding:7px 10px;text-align:right}}
 th{{background:#1f242c;color:#9ab8ff;text-align:center;position:sticky;top:0}}
 td:first-child{{text-align:left}} tr:nth-child(even){{background:#15181d}}
 .tw{{max-height:520px;overflow:auto;border:1px solid #2a2f37;border-radius:10px}}
 figure{{margin:0 0 18px;background:#1a1d23;border:1px solid #2a2f37;border-radius:10px;padding:10px}}
 figure img{{width:100%;display:block;border-radius:6px}}
 figcaption{{font-size:12px;color:#9aa0a6;margin-top:6px}}
 .note{{font-size:12px;color:#9aa0a6;line-height:1.7;border-left:3px solid #3b4252;padding-left:12px;margin-top:10px}}
</style></head><body>
<h1>油红O 切片 · 脂滴面积占比分析</h1>
<div class="sub">分母口径: <b>整张图面积</b> ｜ 共 {len(df)} 张图像</div>
<div class="kpi">
  <div class="card"><div class="v">{mean_all:.2f}%</div><div class="l">平均脂滴面积占比</div></div>
  <div class="card"><div class="v">{df['脂滴面积占比_全图_%'].min():.2f}%</div><div class="l">最低</div></div>
  <div class="card"><div class="v">{df['脂滴面积占比_全图_%'].max():.2f}%</div><div class="l">最高</div></div>
  <div class="card"><div class="v">{int(df['脂滴个数'].sum())}</div><div class="l">检出脂滴总数</div></div>
</div>
<h2>分析参数</h2><ul class="params">{pstr}</ul>
<h2>汇总图</h2>
<figure><img src="summary.png" alt="summary"><figcaption>占比柱状图 / 分组箱线图 / 脂滴计数</figcaption></figure>
<h2>逐图定量表</h2><div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>
<h2>分割校验面板</h2>
<div class="note">每张图从左到右: 原图 → Lab a* 通道 (红色越亮越红) → 二值掩膜 → 叠加结果(黄色=判为脂滴)。
 请重点核对掩膜是否吃进了非脂滴区域(如红色标注线、边缘色差)，或漏掉了淡染脂滴。</div>
{imgs}
</body></html>"""
    out_html.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------- 主流程
def collect_images(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    files = sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXT)
    return files


def dbg_eff_thr_note(args, mode: str) -> str:
    if mode == "fixed":
        return f"红度 R-max(G,B) > {args.thr if args.thr is not None else 40} (全组统一, 组间可比)"
    return f"{args.thr if args.thr else '自动 (Otsu)'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="fixed", choices=["auto", "fixed", "mix", "a", "red", "sat"])
    ap.add_argument("--thr", type=float, default=None,
                    help="fixed 模式下=红度绝对阈值(默认40), 其他模式=对应通道阈值")
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--watershed", action="store_true")
    ap.add_argument("--tighten", type=int, default=0,
                    help="掩膜腐蚀像素数, 抵消红色外扩 (合成真值校准: 1 较准, 默认 0)")
    ap.add_argument("--group-from-name", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = collect_images(inp)
    if not files:
        print("未找到图像文件"); sys.exit(1)

    mode = "fixed" if args.mode == "auto" else args.mode
    rows, panels = [], []
    for i, f in enumerate(files, 1):
        try:
            m, dbg, mask, img = analyze_one(f, mode, args.thr, args.min_area,
                                            args.watershed, args.group_from_name,
                                            args.tighten)
        except Exception as e:
            print(f"[跳过] {f.name}: {e}")
            continue
        rows.append(m.as_row())
        png = out / f"panel_{i:03d}_{Path(f.name).stem}.png"
        make_panel(img, mask, dbg, m, png)
        panels.append(png)
        print(f"[{i}/{len(files)}] {f.name}  占比(全图)={m.ratio_image*100:.3f}%  "
              f"个数={m.droplet_count}  阈值={dbg['eff_thr']:.0f} ({dbg['mode']})")

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    make_summary(df, out / "summary.png")
    make_report_html(df, panels, out / "report.html", {
        "分割模式": mode,
        "统一阈值": dbg_eff_thr_note(args, mode),
        "最小颗粒面积(px)": args.min_area,
        "边缘收敛(腐蚀px)": args.tighten,
        "分水岭拆分": "开" if args.watershed else "关",
        "分母口径": "整张图面积 (同时给出组织面积口径作对照)",
    })

    print("\n=== 汇总 (分母=整张图) ===")
    print(df[["图像", "分组", "脂滴面积占比_全图_%", "脂滴面积占比_组织_%", "脂滴个数"]]
          .to_string(index=False))
    print(f"\n均值: {df['脂滴面积占比_全图_%'].mean():.3f}%   "
          f"SD: {df['脂滴面积占比_全图_%'].std():.3f}%")
    print(f"\n输出目录: {out}")


if __name__ == "__main__":
    main()
