# 合并数据集（merge_dataset.py）

把一批新的 YOLO 标注数据并入已有训练集。新数据里的类别可以是已有类别，也可以是新类别。脚本在 `util/merge_dataset.py`，Python 3.8 起可用（`yolov8_env` 可直接运行），只依赖 PyYAML。

## 先看三条原则

1. **按类名合并，不按 id 合并。** 不同来源的数据里，同一个类别的 id 往往不一样。脚本用类名对齐到目标数据集的 id，所以 SRC 里 `0=Triceratops` 会被正确改写成 DEST 里的 `1`。
2. **默认只预演。** 不加 `--apply` 就只打印计划，不写任何文件。每次先看计划再执行。
3. **每次执行都能撤销。** 写入前会备份 `data.yaml` 并记录清单，`rollback` 可以还原。

## 目录要求

SRC 和 DEST 都是 ultralytics 结构：

```text
<root>/images/{train,val}/*.jpg
<root>/labels/{train,val}/*.txt
<root>/data.yaml            # DEST 必须有；SRC 有的话用来读类名
```

SRC 只有平铺的 `images/*.jpg` 时，用 `--flat-split train|val` 指定全部归入哪个集。

## 快速开始（你的增强数据）

```bash
cd /home/xiaozhang/code/ultralytics
PY=/home/xiaozhang/miniconda3/envs/yolov8_env/bin/python
SRC=/home/xiaozhang/code/PictureAugment/data/augmented
DEST=dataset/dataset/zhou/yolo_dataset

# 1. 预演（不改文件）
$PY util/merge_dataset.py merge $SRC $DEST --splits train

# 2. 确认计划无误后执行
$PY util/merge_dataset.py merge $SRC $DEST --splits train --apply

# 3. 检查结果
$PY util/merge_dataset.py check $DEST
```

### 为什么只并入 train

`augmented` 是现有 332 张图每张做 7 次增强得到的（265 张 train、67 张 val），没有新的原图。

- train 的增强图：可以并入，作用是扩充训练样本。
- val 的增强图：**不要并入 val。** 它们是 val 里原图的变体，验证集会混进近似重复的图，指标虚高，也不提供独立的验证信号。并入 train 更不行：这些图的原图在 val 里，会造成泄漏。脚本会拦住这两种情况，除非加 `--allow-leak`。

用 `--splits train` 之后，val 保持原来的 67 张原图，前后训练的指标可以直接对比。

两点预期，我没有实测：

- 训练集从 265 张变成约 2120 张（约 8 倍），每个 epoch 时间相应变长，复制需要约 12.7 GiB 磁盘。
- ultralytics 训练时本来就有在线增强（马赛克、翻转、色彩扰动等），离线增强带来的提升可能有限。建议合并前后用同样参数各训一次，比较 val 上的 mAP，再决定保留与否。

## 常见场景

| 场景 | 命令 |
|---|---|
| 新数据是已有类别，类名与 DEST 一致 | `merge SRC DEST --apply` |
| SRC 的 id 顺序和 DEST 不同 | 同上，脚本按类名自动重映射 |
| 新数据里有 DEST 没有的类别 | 加 `--add-new-classes`，新类追加在末尾，已有 id 不变 |
| 类名叫法不同（如 `小盗龙` 与 `Microraptor`） | `--map 小盗龙=Microraptor`，可重复 |
| SRC 没有 data.yaml | `--src-names Microraptor,Triceratops`（按 id 顺序），或传 classes.txt 路径 |
| 想和 DEST 的文件名区分开 | `--prefix batch2_` |
| 有些图片没有标注文件（纯背景图） | `--allow-background`，生成空标注 |
| 个别标注有问题，不想整批失败 | `--skip-bad`，跳过坏样本并统计数量 |
| SRC 没有 train/val 子目录 | `--flat-split train` |

## 脚本会拦住的问题

- 类别不在 DEST 里，且没加 `--add-new-classes` 或 `--map`。
- 标注格式不对：不是 5 列、类别 id 越界、坐标没归一化到 0–1。
- 缺少标注文件（没加 `--allow-background`）。
- DEST 已有同名文件但内容不同（防止覆盖）。内容完全相同的会被跳过，所以**重复执行同一次合并是安全的**。
- train/val 同源泄漏。同源判断是文件名去掉 `_augN` 后缀得到原图名，同一原图的变体视为一组；命名规则不同时用 `--group-re` 改正则。
- 磁盘空间不足（复制后至少要剩 1 GiB）。

写入中途失败（如磁盘写满）会自动删除本次已复制的文件并还原 `data.yaml`。

## 撤销

```bash
$PY util/merge_dataset.py rollback $DEST            # 预演：看会删什么
$PY util/merge_dataset.py rollback $DEST --apply    # 执行
```

- 只能按“后合并的先撤销”的顺序撤销，避免还原旧 `data.yaml` 时丢掉后面合并新增的类别。
- 记录保存在 `<DEST>/.merge_history/`，包含清单和 `data.yaml` 备份。
- 撤销只删除本次合并新增的文件，不会动原有文件。

## 合并后要做的事

### 只有已有类别

直接重新训练，沿用你之前的训练命令，确认 `data=` 指向 DEST 的 `data.yaml`。`labels/*.cache` 无需手动删，ultralytics 发现文件变化会重建（预期行为，我没有在这个数据集上实测）。

### 新增了类别

1. 类别数 `nc` 变了，需要重新训练。可以从预训练权重（如 `yolov8n.pt`）或上一次的 `best.pt` 出发；用 `best.pt` 时，检测头的形状对不上，ultralytics 会重新初始化这部分，其余层沿用（预期行为，未实测）。
2. **旧图里没有标新类别。** 如果旧图里其实有新类别的目标但没标，会被当作背景，拖累新类的效果。必要时回去补标。
3. 部署服务（`yolo_llm_code`）要同步改，否则识别到新类别也回答不了：
   - 把新的 `best.pt` 放到 `yolo_llm_code/model/best.pt`。
   - 在 `data/exhibit_map.json` 加 `"新类名": "展品名"`。
   - 在 `data/knowledge/` 加 `展品名.txt` 资料，重启服务重建索引。

   未在 `exhibit_map.json` 里的类别，图片识别后会被当作“未识别到已收录展品”。

## 检查命令

```bash
$PY util/merge_dataset.py check $DEST
```

输出各集的图片数、空标注数、各类标注框数，并检查：类别 id 是否越界、标注格式、有无没有图片的孤立标注、train/val 是否同源。有错误时退出码为 1。

## 测试

```bash
python -m pytest util/test_merge_dataset.py -q
```

测试只用临时目录里的假数据，覆盖类名重映射、新类别、别名、重复合并、同源泄漏、坏标注、中途失败自动撤销和回滚。
