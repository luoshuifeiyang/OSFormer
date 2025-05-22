import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# 数据集路径
DATASET_ROOT = "/path/to/your_dataset"  # 修改为你的数据集路径

# 注册训练集
register_coco_instances(
    "custom_train",
    {},
    os.path.join(DATASET_ROOT, "annotations/instances_train.json"),
    os.path.join(DATASET_ROOT, "train")
)

# 注册验证集
register_coco_instances(
    "custom_val",
    {},
    os.path.join(DATASET_ROOT, "annotations/instances_val.json"),
    os.path.join(DATASET_ROOT, "val")
)

# 注册测试集
register_coco_instances(
    "custom_test",
    {},
    os.path.join(DATASET_ROOT, "annotations/instances_test.json"),
    os.path.join(DATASET_ROOT, "test")
)

# 设置类别信息
# 根据你的数据集修改类别名称
CUSTOM_CATEGORIES = [
    "category1", "category2", "category3", # 替换为你的类别
    # ... 添加所有类别
]

# 更新metadata
MetadataCatalog.get("custom_train").thing_classes = CUSTOM_CATEGORIES
MetadataCatalog.get("custom_val").thing_classes = CUSTOM_CATEGORIES
MetadataCatalog.get("custom_test").thing_classes = CUSTOM_CATEGORIES

print("Dataset registered successfully!")
print(f"Training samples: {len(DatasetCatalog.get('custom_train'))}")
print(f"Validation samples: {len(DatasetCatalog.get('custom_val'))}")
print(f"Test samples: {len(DatasetCatalog.get('custom_test'))}")