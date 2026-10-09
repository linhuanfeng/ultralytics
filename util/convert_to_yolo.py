"""把按类别分目录的 YOLO 标注数据集转换成可直接训练的 ultralytics 数据集。

输入结构 (每个类别一个子目录, 标注里的类别 id 在各目录内各自从 0 开始):
    mark/images/<类别名>/*.jpg
    mark/labels/<类别名>/*.txt  (+ classes.txt)

输出结构:
    <out>/images/{train,val}/*.jpg
    <out>/labels/{train,val}/*.txt   (类别 id 已按全局类别表重新映射)
    <out>/data.yaml

用法:
    python convert_to_yolo.py
    python convert_to_yolo.py --src zhou/mark --out zhou/yolo_dataset --val-ratio 0.2
"""

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SRC = Path(__file__).parent / "dataset" / "zhou" / "mark"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="含 images/ 和 labels/ 的源目录")
    p.add_argument("--out", type=Path, default=None, help="输出目录, 默认为 <src>/../yolo_dataset")
    p.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例 (按类别分层抽样)")
    p.add_argument("--seed", type=int, default=0, help="随机种子, 保证划分可复现")
    p.add_argument("--link", action="store_true", help="用软链接代替复制图片, 节省磁盘")
    p.add_argument("--force", action="store_true", help="输出目录已存在时先清空")
    return p.parse_args()


def read_label(path: Path, new_id: int):
    """读取标注文件, 校验格式并把类别 id 改写为 new_id。返回新的标注行列表。"""
    rows = []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}:{ln} 应有 5 列, 实际 {len(parts)} 列")
        x, y, w, h = map(float, parts[1:])
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{path}:{ln} 坐标越界: {parts[1:]}")
        rows.append(f"{new_id} {' '.join(parts[1:])}")
    if not rows:
        raise ValueError(f"{path} 是空标注文件")
    return rows


def main():
    args = parse_args()
    src = args.src.resolve()
    out = (args.out or src.parent / "yolo_dataset").resolve()
    img_root, lbl_root = src / "images", src / "labels"

    if not img_root.is_dir() or not lbl_root.is_dir():
        raise SystemExit(f"{src} 下找不到 images/ 或 labels/")
    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio 必须在 (0, 1) 之间")

    # 全局类别表: 以 images/ 下的子目录名为准, 按字母序固定 id
    classes = sorted(d.name for d in img_root.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"{img_root} 下没有类别子目录")

    # 收集样本并校验
    samples = {}  # 类别 -> [(图片路径, 标注路径)]
    seen = {}
    for cid, cls in enumerate(classes):
        cls_file = lbl_root / cls / "classes.txt"
        if cls_file.exists():
            names = [n.strip() for n in cls_file.read_text().splitlines() if n.strip()]
            if names != [cls]:
                print(f"[警告] {cls_file} 内容 {names} 与目录名 {cls!r} 不一致, 以目录名为准")

        items = []
        for img in sorted(p for p in (img_root / cls).iterdir() if p.suffix.lower() in IMG_EXTS):
            lbl = lbl_root / cls / f"{img.stem}.txt"
            if not lbl.exists():
                raise SystemExit(f"图片缺少标注: {img}")
            if img.name in seen:
                raise SystemExit(f"文件名重复: {img} 与 {seen[img.name]}")
            seen[img.name] = img
            read_label(lbl, cid)  # 仅校验, 失败直接报错
            items.append((img, lbl))
        if not items:
            raise SystemExit(f"类别 {cls} 没有图片")
        samples[cls] = items

    # 准备输出目录
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} 已存在, 加 --force 清空后重建")
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    # 按类别分层划分并写出
    rng = random.Random(args.seed)
    stats = {}
    for cid, cls in enumerate(classes):
        items = samples[cls][:]
        rng.shuffle(items)
        n_val = max(1, round(len(items) * args.val_ratio))
        splits = {"val": items[:n_val], "train": items[n_val:]}
        stats[cls] = {k: len(v) for k, v in splits.items()}
        for split, pairs in splits.items():
            for img, lbl in pairs:
                dst_img = out / "images" / split / img.name
                if args.link:
                    dst_img.symlink_to(img)
                else:
                    shutil.copy2(img, dst_img)
                (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(read_label(lbl, cid)) + "\n")

    # data.yaml (手写, 避免依赖 PyYAML)
    lines = [f"path: {out}", "train: images/train", "val: images/val", "", f"nc: {len(classes)}", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(classes)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")

    print(f"输出目录: {out}")
    for cid, cls in enumerate(classes):
        s = stats[cls]
        print(f"  [{cid}] {cls}: train={s['train']} val={s['val']}")
    print(f"data.yaml: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
