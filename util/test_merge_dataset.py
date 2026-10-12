"""merge_dataset.py 的测试: 只用临时目录里的假数据,不碰真实数据集。

运行: python -m pytest util/test_merge_dataset.py -q
"""
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import merge_dataset as md  # noqa: E402


def make_ds(root, names, files, with_yaml=True):
    """files: [(split, stem, [class ids])];图片内容用 stem 填充,方便区分不同文件。"""
    root = Path(root)
    for split, stem, ids in files:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "images" / split / (stem + ".jpg")).write_bytes(stem.encode())
        lines = ["{} 0.5 0.5 0.2 0.2".format(i) for i in ids]
        (root / "labels" / split / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    if with_yaml:
        (root / "data.yaml").write_text(yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val",
             "nc": len(names), "names": dict(enumerate(names))}, allow_unicode=True))
    return root


def run(*argv):
    return md.main([str(a) for a in argv])


def names_of(root):
    return md.names_from_yaml(yaml.safe_load((Path(root) / "data.yaml").read_text())["names"], "t")


def count(root, split):
    return len(list((Path(root) / "images" / split).glob("*.jpg")))


@pytest.fixture
def dest(tmp_path):
    return make_ds(tmp_path / "dest", ["Microraptor", "Triceratops"],
                   [("train", "a", [0]), ("train", "b", [1]), ("val", "c", [0])])


def test_dry_run_changes_nothing(tmp_path, dest, capsys):
    src = make_ds(tmp_path / "src", ["Microraptor", "Triceratops"], [("train", "x", [0])])
    before = sorted(p.name for p in dest.rglob("*"))
    assert run("merge", src, dest) == 0
    assert sorted(p.name for p in dest.rglob("*")) == before
    assert "预演" in capsys.readouterr().out


def test_existing_classes_merged_by_name_not_by_id(tmp_path, dest):
    # SRC 的 id 顺序和 DEST 相反: 0=Triceratops, 1=Microraptor,必须按类名重映射
    src = make_ds(tmp_path / "src", ["Triceratops", "Microraptor"],
                  [("train", "x", [0]), ("train", "y", [1])])
    assert run("merge", src, dest, "--apply") == 0
    assert (dest / "labels/train/x.txt").read_text().split()[0] == "1"  # Triceratops 在 DEST 是 1
    assert (dest / "labels/train/y.txt").read_text().split()[0] == "0"
    assert names_of(dest) == ["Microraptor", "Triceratops"]
    assert count(dest, "train") == 4


def test_new_class_needs_flag_then_is_appended(tmp_path, dest, capsys):
    src = make_ds(tmp_path / "src", ["Velociraptor"], [("train", "v", [0])])
    assert run("merge", src, dest, "--apply") == 2
    assert "--add-new-classes" in capsys.readouterr().err
    assert count(dest, "train") == 2 and names_of(dest) == ["Microraptor", "Triceratops"]
    assert run("merge", src, dest, "--add-new-classes", "--apply") == 0
    assert names_of(dest) == ["Microraptor", "Triceratops", "Velociraptor"]  # 旧 id 不变
    assert (dest / "labels/train/v.txt").read_text().split()[0] == "2"
    cfg = yaml.safe_load((dest / "data.yaml").read_text())
    assert cfg["nc"] == 3 and cfg["train"] == "images/train"  # 其它键保留


def test_alias_maps_to_existing_class(tmp_path, dest):
    src = make_ds(tmp_path / "src", ["小盗龙"], [("train", "z", [0])])
    assert run("merge", src, dest, "--map", "小盗龙=Microraptor", "--apply") == 0
    assert names_of(dest) == ["Microraptor", "Triceratops"]
    assert (dest / "labels/train/z.txt").read_text().split()[0] == "0"


def test_rerun_is_idempotent_and_content_conflict_fails(tmp_path, dest, capsys):
    src = make_ds(tmp_path / "src", ["Microraptor"], [("train", "x", [0])])
    assert run("merge", src, dest, "--apply") == 0
    assert run("merge", src, dest, "--apply") == 0  # 内容相同 -> 跳过
    assert count(dest, "train") == 3
    (src / "images/train/x.jpg").write_bytes(b"different")
    assert run("merge", src, dest, "--apply") == 2  # 同名不同内容 -> 拒绝覆盖
    assert (dest / "images/train/x.jpg").read_bytes() == b"x"
    assert run("merge", src, dest, "--prefix", "s2_", "--apply") == 0
    assert (dest / "images/train/s2_x.jpg").exists()


def test_augmented_val_leak_blocked_and_train_only_ok(tmp_path, dest, capsys):
    # DEST val 有原图 c;SRC 的 train 带来 c 的增强版 -> 同源泄漏
    src = make_ds(tmp_path / "src", ["Microraptor"],
                  [("train", "c_aug0", [0]), ("val", "a_aug0", [0])])
    assert run("merge", src, dest, "--apply") == 2
    assert "同源泄漏" in capsys.readouterr().err
    assert count(dest, "train") == 2
    # 只并入 val 里 a 的增强版会违规(a 在 DEST train);两个方向都被拦
    assert run("merge", src, dest, "--splits", "val", "--apply") == 2
    ok = make_ds(tmp_path / "ok", ["Microraptor"], [("train", "a_aug0", [0]), ("train", "b_aug0", [1])],)
    (ok / "data.yaml").unlink()
    assert run("merge", ok, dest, "--src-names", "Microraptor,Triceratops", "--apply") == 0
    assert count(dest, "train") == 4


def test_bad_labels_fail_unless_skip_bad(tmp_path, dest, capsys):
    src = make_ds(tmp_path / "src", ["Microraptor"], [("train", "g", [0]), ("train", "bad", [0])])
    (src / "labels/train/bad.txt").write_text("0 1.5 0.5 0.2 0.2\n")
    assert run("merge", src, dest, "--apply") == 2
    assert count(dest, "train") == 2
    assert run("merge", src, dest, "--skip-bad", "--apply") == 0
    assert count(dest, "train") == 3 and not (dest / "images/train/bad.jpg").exists()


def test_missing_label_needs_allow_background(tmp_path, dest):
    src = make_ds(tmp_path / "src", ["Microraptor"], [("train", "bg", [0])])
    (src / "labels/train/bg.txt").unlink()
    assert run("merge", src, dest, "--apply") == 2
    assert run("merge", src, dest, "--allow-background", "--apply") == 0
    assert (dest / "labels/train/bg.txt").read_text() == ""


def test_flat_layout_requires_flat_split(tmp_path, dest):
    src = tmp_path / "flat"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir()
    (src / "images/f.jpg").write_bytes(b"f")
    (src / "labels/f.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    assert run("merge", src, dest, "--src-names", "Microraptor", "--apply") == 2
    assert run("merge", src, dest, "--src-names", "Microraptor", "--flat-split", "val", "--apply") == 0
    assert (dest / "images/val/f.jpg").exists()


def test_rollback_restores_files_and_classes(tmp_path, dest):
    snapshot = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if ".merge_history" not in str(p))
    src = make_ds(tmp_path / "src", ["Velociraptor"], [("train", "v", [0])])
    assert run("merge", src, dest, "--add-new-classes", "--apply") == 0
    assert run("rollback", dest) == 0 and (dest / "images/train/v.jpg").exists()  # 预演不删
    assert run("rollback", dest, "--apply") == 0
    after = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if ".merge_history" not in str(p))
    assert after == snapshot and names_of(dest) == ["Microraptor", "Triceratops"]
    assert run("rollback", dest, "--apply") == 2  # 已无可撤销记录


def test_failure_midway_undoes_partial_copy(tmp_path, dest, monkeypatch):
    src = make_ds(tmp_path / "src", ["Velociraptor"], [("train", "p1", [0]), ("train", "p2", [0])])
    real, calls = shutil.copyfile, []

    def flaky(a, b, **kw):
        # copy2(备份/恢复 data.yaml)内部也会走 copyfile,只对图片目标计数
        if "images" in str(b):
            calls.append(b)
            if len(calls) == 2:
                raise OSError("disk error")
        return real(a, b, **kw)

    monkeypatch.setattr(md.shutil, "copyfile", flaky)
    with pytest.raises(OSError):
        run("merge", src, dest, "--add-new-classes", "--apply")
    assert count(dest, "train") == 2 and not list((dest / "labels/train").glob("p*"))
    assert names_of(dest) == ["Microraptor", "Triceratops"]


def test_check_detects_leak_and_bad_class_id(tmp_path, capsys):
    ds = make_ds(tmp_path / "ds", ["A", "B"],
                 [("train", "o_aug0", [0]), ("val", "o_aug1", [1]), ("val", "q", [5])])
    assert run("check", ds) == 1
    out = capsys.readouterr().out
    assert "同源" in out and "超出范围" in out
    clean = make_ds(tmp_path / "clean", ["A", "B"], [("train", "t1", [0]), ("val", "v1", [1])])
    assert run("check", clean) == 0
