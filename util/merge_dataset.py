"""把一批 YOLO 标注数据合并进已有的训练数据集(类别可以已存在,也可以是新增的)。

子命令:
    merge     把 SRC 合并进 DEST。默认只预演(dry-run)并打印计划,加 --apply 才真正写入
    check     检查数据集: 图片/标注配对、类别 id 范围、各类数量、train/val 同源泄漏
    rollback  撤销最近(或指定)一次 merge。默认只预演,加 --apply 才真正删除

SRC 与 DEST 都是 ultralytics 目录结构:
    <root>/images/{train,val}/*.jpg
    <root>/labels/{train,val}/*.txt
    DEST 另需 <root>/data.yaml

用法见同目录 MERGE_DATASET.md。兼容 Python 3.8(训练环境 yolov8_env)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")
# 增强图文件名形如 <原图名>_aug3,去掉后缀得到原图名;同一原图的所有变体视为一组
DEFAULT_GROUP_RE = r"(_aug\d+)+$"
HISTORY = ".merge_history"
FREE_SPACE_MARGIN = 1 << 30  # 复制后至少还要剩 1GiB


class MergeError(Exception):
    """可预期的数据/参数问题: 打印原因后以非零状态退出,不抛堆栈。"""


def read_dest_yaml(dest: Path):
    """读取 DEST/data.yaml,返回 (原始字典, 类名列表[按 id 排序])。"""
    path = dest / "data.yaml"
    if not path.exists():
        raise MergeError("DEST 缺少 data.yaml: {}".format(path))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data, names_from_yaml(data.get("names"), path)


def names_from_yaml(names, where) -> list:
    if isinstance(names, dict):
        try:
            items = sorted((int(k), str(v)) for k, v in names.items())
        except ValueError:
            raise MergeError("{} 的 names 键必须是整数 id".format(where))
        if [k for k, _ in items] != list(range(len(items))):
            raise MergeError("{} 的类别 id 必须从 0 连续编号".format(where))
        return [v for _, v in items]
    if isinstance(names, list):
        return [str(v) for v in names]
    raise MergeError("{} 里没有可识别的 names".format(where))


def parse_name_spec(spec: str) -> list:
    """--src-names 可以是 data.yaml / classes.txt 路径,或逗号分隔的类名(按 id 顺序)。"""
    p = Path(spec)
    if p.is_file():
        if p.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return names_from_yaml(data.get("names"), p)
        return [s.strip() for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
    if p.suffix in (".yaml", ".yml", ".txt"):
        raise MergeError("--src-names 指定的文件不存在: {}".format(spec))
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise MergeError("--src-names 为空")
    return names


def find_src_names(src: Path, spec):
    """返回 (类名列表, 来源说明)。未指定时在 SRC 及其上一级找 data.yaml / classes.txt。"""
    if spec:
        return parse_name_spec(spec), "--src-names"
    for d in (src, src.parent):
        for fname in ("data.yaml", "classes.txt"):
            if (d / fname).is_file():
                return parse_name_spec(str(d / fname)), str(d / fname)
    raise MergeError("找不到 SRC 的类别名,请用 --src-names 指定(data.yaml / classes.txt / 逗号分隔类名)")


def build_class_map(src_names, dest_names, renames, allow_new):
    """按类名把 SRC 的类别 id 映射到 DEST 的 id。同名即同类;新名字追加在末尾。

    返回 (src_id -> dest_id, 合并后的类名列表, 新增的类名列表)。
    """
    final = list(dest_names)
    mapping, added = {}, []
    for sid, name in enumerate(src_names):
        name = renames.get(name, name)
        if name not in final:
            if not allow_new:
                raise MergeError(
                    "SRC 的类别 '{}' 不在 DEST 的类别里: {}\n"
                    "  - 是新类别: 加 --add-new-classes\n"
                    "  - 是已有类别的别名: 加 --map {}=<DEST 里的类名>".format(name, dest_names, name))
            final.append(name)
            added.append(name)
        mapping[sid] = final.index(name)
    return mapping, final, added


def list_split(root: Path, split: str) -> tuple:
    """返回 (图片列表, 标注 stem 集合)。目录不存在视为空。"""
    img_dir, lab_dir = root / "images" / split, root / "labels" / split
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS) if img_dir.is_dir() else []
    labs = {p.stem for p in lab_dir.glob("*.txt")} if lab_dir.is_dir() else set()
    return imgs, labs


def check_label(path: Path, nc: int):
    """校验一个标注文件。返回 (错误列表, 各行类别 id 列表)。空文件合法(背景图)。"""
    errors, ids = [], []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append("{}:{} 应有 5 列,实际 {}".format(path.name, ln, len(parts)))
            continue
        try:
            cid = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append("{}:{} 含非数字".format(path.name, ln))
            continue
        if not 0 <= cid < nc:
            errors.append("{}:{} 类别 id {} 超出范围 [0,{})".format(path.name, ln, cid, nc))
        elif not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append("{}:{} 坐标未归一化到 0-1".format(path.name, ln))
        else:
            ids.append(cid)
    return errors, ids


def group_of(stem: str, group_re: str) -> str:
    return re.sub(group_re, "", stem)


def same_file(a: Path, b: Path) -> bool:
    """大小相同再比 sha256,避免对每个文件都读全文。"""
    if a.stat().st_size != b.stat().st_size:
        return False
    return sha256(a) == sha256(b)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_renames(pairs) -> dict:
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise MergeError("--map 格式应为 旧名=新名,收到: {}".format(item))
        old, new = item.split("=", 1)
        out[old.strip()] = new.strip()
    return out


def source_pairs(src: Path, splits, flat_split):
    """产出 (目标 split, 图片路径, 标注路径或 None, 原 stem)。

    SRC 有 images/<split>/ 子目录时按子目录;只有平铺的 images/*.jpg 时必须用 --flat-split 指定去向。
    """
    layout_split = any((src / "images" / s).is_dir() for s in SPLITS)
    if layout_split:
        for s in splits:
            imgs, _ = list_split(src, s)
            for img in imgs:
                lab = src / "labels" / s / (img.stem + ".txt")
                yield s, img, (lab if lab.exists() else None)
        return
    if not (src / "images").is_dir():
        raise MergeError("SRC 下没有 images 目录: {}".format(src))
    if not flat_split:
        raise MergeError("SRC 的 images 下没有 train/val 子目录,请用 --flat-split train|val 指定全部归入哪个集")
    for img in sorted(p for p in (src / "images").iterdir() if p.suffix.lower() in IMG_EXTS):
        lab = src / "labels" / (img.stem + ".txt")
        yield flat_split, img, (lab if lab.exists() else None)


def remap_label(path, cmap) -> str:
    """把标注每行第一列的类别 id 换成 DEST 的 id,其余列原样保留。path 为 None 表示背景图。"""
    if path is None:
        return ""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            parts[0] = str(cmap[int(parts[0])])
            lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")


def index_dest(dest: Path) -> dict:
    return {s: {p.stem: p for p in list_split(dest, s)[0]} for s in SPLITS}


def plan_merge(args) -> dict:
    """只读检查并生成合并计划,不写任何文件。有问题时抛 MergeError。"""
    src, dest = Path(args.src).resolve(), Path(args.dest).resolve()
    if not src.is_dir():
        raise MergeError("SRC 不是目录: {}".format(src))
    if src == dest:
        raise MergeError("SRC 与 DEST 不能是同一个目录")
    yaml_data, dest_names = read_dest_yaml(dest)
    src_names, names_from = find_src_names(src, args.src_names)
    cmap, final, added = build_class_map(src_names, dest_names, parse_renames(args.map), args.add_new_classes)
    dest_idx, prefix = index_dest(dest), args.prefix or ""
    items, problems, seen, skipped_bad = [], [], set(), 0
    for split, img, lab in source_pairs(src, tuple(args.splits), args.flat_split):
        stem = prefix + img.stem
        errs = []
        if (split, stem.lower()) in seen:
            errs.append("SRC 内同一集出现重名(仅扩展名不同): {}".format(img.name))
        seen.add((split, stem.lower()))
        if lab is None and not args.allow_background:
            errs.append("缺少标注文件: {}(背景图请加 --allow-background)".format(img.name))
        elif lab is not None:
            lerrs, _ = check_label(lab, len(src_names))
            errs.extend(lerrs)
        if errs:
            if args.skip_bad:
                skipped_bad += 1
            else:
                problems.extend(errs)
            continue
        text = remap_label(lab, cmap)
        status = "new"
        old = dest_idx[split].get(stem)
        if old is not None:
            dlab = dest / "labels" / split / (stem + ".txt")
            same = old.suffix.lower() == img.suffix.lower() and same_file(img, old) \
                and dlab.exists() and dlab.read_text(encoding="utf-8") == text
            if not same:
                problems.append("DEST 已有同名但内容不同的文件: {}/{}(可用 --prefix 区分来源)".format(split, old.name))
                continue
            status = "skip"
        items.append({"split": split, "img": img, "stem": stem, "text": text, "status": status})
    if problems:
        head = "\n  ".join(problems[:10])
        more = "\n  ...共 {} 个问题".format(len(problems)) if len(problems) > 10 else ""
        raise MergeError("合并前检查未通过:\n  " + head + more + "\n(坏样本可加 --skip-bad 跳过)")
    notes, leaks = [], []
    inc = {s: {group_of(i["stem"], args.group_re) for i in items if i["split"] == s} for s in SPLITS}
    old = {s: {group_of(k, args.group_re) for k in dest_idx[s]} for s in SPLITS}
    for a, b in (("train", "val"), ("val", "train")):
        n = len(inc[a] & old[b])
        if n:
            leaks.append("新增的 {} 集有 {} 个原图组,其变体已在 DEST 的 {} 集".format(a, n, b))
    if len(inc["train"] & inc["val"]):
        leaks.append("SRC 内有 {} 个原图组同时出现在 train 和 val".format(len(inc["train"] & inc["val"])))
    if leaks and not args.allow_leak:
        raise MergeError("检测到 train/val 同源泄漏(会让验证指标虚高):\n  " + "\n  ".join(leaks)
                         + "\n确认无误可加 --allow-leak")
    notes.extend("[泄漏,已按 --allow-leak 放行] " + m for m in leaks)
    for s in SPLITS:
        n = len(inc[s] & old[s])
        if n:
            notes.append("新增的 {} 集有 {} 个原图组在 DEST 的 {} 集里已有原图或其它变体;".format(s, n, s)
                         + ("增强图并入 val 不提供独立验证信号,建议 --splits train" if s == "val" else "属于增强扩充,可正常训练"))
    new_items = [i for i in items if i["status"] == "new"]
    need = sum(i["img"].stat().st_size for i in new_items)
    free = shutil.disk_usage(str(dest)).free
    if need + FREE_SPACE_MARGIN > free:
        raise MergeError("磁盘空间不足: 需要 {:.1f}GiB,可用 {:.1f}GiB".format(need / 2**30, free / 2**30))
    counts = {s: Counter(int(ln.split()[0]) for i in new_items if i["split"] == s for ln in i["text"].splitlines()) for s in SPLITS}
    return {"src": src, "dest": dest, "items": items, "new": new_items, "final": final, "added": added,
            "dest_names": dest_names, "src_names": src_names, "names_from": names_from, "cmap": cmap,
            "notes": notes, "need": need, "skipped_bad": skipped_bad, "yaml_data": yaml_data, "counts": counts}


def print_plan(plan: dict, applying: bool) -> None:
    print("SRC : {}".format(plan["src"]))
    print("DEST: {}".format(plan["dest"]))
    print("SRC 类别名来源: {}  ->  {}".format(plan["names_from"], plan["src_names"]))
    print("DEST 现有类别 : {}".format(plan["dest_names"]))
    for sid, did in plan["cmap"].items():
        print("  类别映射: SRC {} ({}) -> DEST {} ({})".format(sid, plan["src_names"][sid], did, plan["final"][did]))
    if plan["added"]:
        print("新增类别(追加在末尾,已有 id 不变): {}".format(plan["added"]))
    for s in SPLITS:
        new = sum(1 for i in plan["new"] if i["split"] == s)
        skip = sum(1 for i in plan["items"] if i["split"] == s and i["status"] == "skip")
        print("  {:5s}: 新增 {} 张,已存在(内容相同,跳过) {} 张,各类标注框 {}".format(
            s, new, skip, dict(sorted(plan["counts"][s].items()))))
    if plan["skipped_bad"]:
        print("  已按 --skip-bad 跳过 {} 个有问题的样本".format(plan["skipped_bad"]))
    print("  需复制 {:.2f}GiB".format(plan["need"] / 2**30))
    for n in plan["notes"]:
        print("注意: " + n)
    print("-> 开始写入" if applying else "-> 这是预演,未改动任何文件。确认无误后加 --apply 执行")


def write_yaml(path: Path, data: dict, names: list) -> None:
    data = dict(data)
    data["names"] = {i: n for i, n in enumerate(names)}
    data["nc"] = len(names)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def apply_plan(plan: dict, args) -> Path:
    dest = plan["dest"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    hist = dest / HISTORY
    hist.mkdir(exist_ok=True)
    backup = hist / "data.yaml.{}.bak".format(stamp)
    shutil.copy2(str(dest / "data.yaml"), str(backup))
    created = []
    try:
        for it in plan["new"]:
            img_dst = dest / "images" / it["split"] / (it["stem"] + it["img"].suffix.lower())
            lab_dst = dest / "labels" / it["split"] / (it["stem"] + ".txt")
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            lab_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(it["img"]), str(img_dst))
            created.append(img_dst)
            lab_dst.write_text(it["text"], encoding="utf-8")
            created.append(lab_dst)
        if plan["added"]:
            write_yaml(dest / "data.yaml", plan["yaml_data"], plan["final"])
    except BaseException:
        for p in created:
            if p.exists():
                p.unlink()
        shutil.copy2(str(backup), str(dest / "data.yaml"))
        print("写入中断,已撤销本次已复制的文件并恢复 data.yaml", file=sys.stderr)
        raise
    manifest = hist / "merge-{}.json".format(stamp)
    manifest.write_text(json.dumps({
        "stamp": stamp, "src": str(plan["src"]), "added_classes": plan["added"],
        "yaml_backup": backup.name, "rolled_back": False,
        "files": [str(p.relative_to(dest)) for p in created],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def cmd_merge(args) -> int:
    plan = plan_merge(args)
    print_plan(plan, args.apply)
    if not plan["new"] and not plan["added"]:
        print("没有需要新增的内容。")
        return 0
    if not args.apply:
        return 0
    manifest = apply_plan(plan, args)
    print("完成,共新增 {} 对图片/标注。撤销用: python merge_dataset.py rollback {} --apply".format(
        len(plan["new"]), plan["dest"]))
    print("清单: {}".format(manifest))
    print("提示: 旧的 labels/*.cache 会被 ultralytics 自动重建;新增类别后请重新训练(见文档)。")
    return 0


def cmd_check(args) -> int:
    root = Path(args.dest).resolve()
    _, names = read_dest_yaml(root)
    errors, warns, groups = [], [], {}
    print("数据集: {}\n类别: {}".format(root, names))
    for s in SPLITS:
        imgs, labs = list_split(root, s)
        stems = {p.stem for p in imgs}
        per_class, empty = Counter(), 0
        for p in imgs:
            lab = root / "labels" / s / (p.stem + ".txt")
            if not lab.exists():
                warns.append("{}/{} 没有标注文件(按背景图处理)".format(s, p.name))
                continue
            errs, ids = check_label(lab, len(names))
            errors.extend(errs)
            per_class.update(ids)
            empty += 0 if ids else 1
        orphan = labs - stems
        if orphan:
            warns.append("{} 有 {} 个标注没有对应图片".format(s, len(orphan)))
        groups[s] = {group_of(t, args.group_re) for t in stems}
        print("  {:5s}: 图片 {} 张,空标注 {},各类标注框 {}".format(
            s, len(imgs), empty, {names[k]: v for k, v in sorted(per_class.items())}))
    leak = groups["train"] & groups["val"]
    if leak:
        errors.append("train/val 有 {} 个原图组同源(例: {}),验证指标会虚高".format(len(leak), sorted(leak)[:3]))
    for w in warns[:10]:
        print("警告: " + w)
    for e in errors[:20]:
        print("错误: " + e)
    print("检查{}".format("未通过: {} 个错误".format(len(errors)) if errors else "通过"))
    return 1 if errors else 0


def cmd_rollback(args) -> int:
    dest = Path(args.dest).resolve()
    hist = dest / HISTORY
    manifests = sorted(hist.glob("merge-*.json")) if hist.is_dir() else []
    pending = [m for m in manifests if not json.loads(m.read_text(encoding="utf-8"))["rolled_back"]]
    if not pending:
        raise MergeError("没有可撤销的合并记录(目录: {})".format(hist))
    path = pending[-1]
    if args.stamp and args.stamp not in path.name:
        # 只能按"后合并的先撤销"的顺序,否则恢复旧 data.yaml 会丢掉后面合并新增的类别
        raise MergeError("只能撤销最近一次未撤销的合并({}),请先撤销它".format(path.name))
    info = json.loads(path.read_text(encoding="utf-8"))
    root = dest.resolve()
    files = []
    for rel in info["files"]:
        f = (dest / rel).resolve()
        if root not in f.parents:  # 清单被改过也不能删到 DEST 之外
            raise MergeError("清单含 DEST 之外的路径,拒绝执行: {}".format(rel))
        files.append(f)
    print("将撤销 {}: 删除 {} 个文件{}".format(path.name, len(files),
          ",并恢复 data.yaml(移除新增类别 {})".format(info["added_classes"]) if info["added_classes"] else ""))
    if not args.apply:
        print("-> 这是预演,未改动任何文件。确认无误后加 --apply 执行")
        return 0
    for f in files:
        if f.exists():
            f.unlink()
    if info["added_classes"]:
        shutil.copy2(str(hist / info["yaml_backup"]), str(dest / "data.yaml"))
    info["rolled_back"] = True
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已撤销。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="合并/检查/撤销 YOLO 数据集")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("merge", help="把 SRC 合并进 DEST(默认预演)")
    m.add_argument("src", help="要并入的数据集根目录(含 images/ labels/)")
    m.add_argument("dest", help="目标数据集根目录(含 data.yaml)")
    m.add_argument("--apply", action="store_true", help="真正写入;不加则只预演")
    m.add_argument("--splits", nargs="+", choices=SPLITS, default=["train", "val"],
                   help="只并入 SRC 的这些集(增强数据建议只用 train)")
    m.add_argument("--flat-split", choices=SPLITS, help="SRC 的 images 下没有 train/val 时,全部归入哪个集")
    m.add_argument("--src-names", help="SRC 的类别名: data.yaml / classes.txt 路径,或按 id 顺序的逗号分隔类名")
    m.add_argument("--map", action="append", metavar="旧名=新名", help="把 SRC 的类名当作 DEST 里的另一个类名,可重复")
    m.add_argument("--add-new-classes", action="store_true", help="允许 SRC 里 DEST 没有的类名作为新类别追加")
    m.add_argument("--prefix", default="", help="给并入的文件名加前缀,避免与 DEST 重名")
    m.add_argument("--allow-background", action="store_true", help="允许没有标注文件的图片(当作无目标的背景图)")
    m.add_argument("--skip-bad", action="store_true", help="跳过标注有问题的样本而不是整体失败")
    m.add_argument("--allow-leak", action="store_true", help="允许 train/val 同源(不推荐)")
    m.add_argument("--group-re", default=DEFAULT_GROUP_RE, help="从文件名 stem 去掉的后缀正则,用于判定同源图(默认去掉 _augN)")
    m.set_defaults(func=cmd_merge)

    c = sub.add_parser("check", help="检查数据集完整性与同源泄漏")
    c.add_argument("dest", help="数据集根目录")
    c.add_argument("--group-re", default=DEFAULT_GROUP_RE)
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("rollback", help="撤销最近一次 merge(默认预演)")
    r.add_argument("dest", help="数据集根目录")
    r.add_argument("--stamp", help="只在文件名含该时间戳的记录里选")
    r.add_argument("--apply", action="store_true")
    r.set_defaults(func=cmd_rollback)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except MergeError as e:
        print("错误: {}".format(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
