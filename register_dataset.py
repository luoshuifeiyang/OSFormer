from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import json


def register_custom_dataset():
    """注册自定义COCO格式数据集"""
    # 数据集根目录 - 修改为你的实际路径
    DATASET_ROOT = "/media/ubuntu/DATA4T/dataset/LIS_paixu/RGB-dark"

    # 检查路径是否存在
    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(f"数据集路径不存在: {DATASET_ROOT}")

    print(f"数据集根目录: {DATASET_ROOT}")

    # 定义数据集配置
    datasets = {
        "custom_train": {
            "json": os.path.join(DATASET_ROOT, "annotations/lis_coco_JPG_train+1.json"),
            "images": os.path.join(DATASET_ROOT, "images")
        },
        "custom_val": {
            "json": os.path.join(DATASET_ROOT, "annotations/lis_coco_JPG_test+1.json"),
            "images": os.path.join(DATASET_ROOT, "images")
        },
        "custom_test": {
            "json": os.path.join(DATASET_ROOT, "annotations/lis_coco_JPG_test+1.json"),
            "images": os.path.join(DATASET_ROOT, "images")
        }
    }

    # 注册数据集
    registered_datasets = []
    for dataset_name, paths in datasets.items():
        json_path = paths["json"]
        images_path = paths["images"]

        if os.path.exists(json_path) and os.path.exists(images_path):
            try:
                register_coco_instances(dataset_name, {}, json_path, images_path)
                print(f"✓ {dataset_name} 注册成功")
                print(f"  - 注解文件: {json_path}")
                print(f"  - 图像目录: {images_path}")
                registered_datasets.append(dataset_name)
            except Exception as e:
                print(f"✗ {dataset_name} 注册失败: {e}")
        else:
            print(f"✗ {dataset_name} 路径不存在:")
            if not os.path.exists(json_path):
                print(f"  - 注解文件不存在: {json_path}")
            if not os.path.exists(images_path):
                print(f"  - 图像目录不存在: {images_path}")

    if not registered_datasets:
        raise RuntimeError("没有成功注册任何数据集")

    # 从注解文件中自动获取类别信息
    train_json = datasets["custom_train"]["json"]
    thing_classes = []
    num_classes = 0

    try:
        with open(train_json, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # 提取类别信息
        categories = coco_data.get('categories', [])
        if not categories:
            print("警告: 注解文件中没有找到类别信息")
            # 使用默认类别（根据你的实际情况修改）
            thing_classes = ["bicycle", "chair", "diningtable", "bottle", "motorbike", "car", "tvmonitor", "bus"]
        else:
            # 按照category_id排序，确保类别顺序正确
            categories = sorted(categories, key=lambda x: x.get('id', 0))
            thing_classes = [cat['name'] for cat in categories]

        num_classes = len(thing_classes)

        print(f"\n类别信息:")
        print(f"类别数量: {num_classes}")
        print(f"类别名称: {thing_classes}")

        # 设置元数据
        for dataset_name in registered_datasets:
            try:
                metadata = MetadataCatalog.get(dataset_name)
                metadata.thing_classes = thing_classes
                metadata.evaluator_type = "coco"
                print(f"✓ {dataset_name} 元数据设置成功")
            except Exception as e:
                print(f"✗ {dataset_name} 元数据设置失败: {e}")

    except Exception as e:
        print(f"警告: 无法读取类别信息: {e}")
        # 使用默认类别
        thing_classes = ["bicycle", "chair", "diningtable", "bottle", "motorbike", "car", "tvmonitor", "bus"]
        num_classes = len(thing_classes)

        for dataset_name in registered_datasets:
            try:
                MetadataCatalog.get(dataset_name).thing_classes = thing_classes
            except Exception as e:
                print(f"设置默认类别失败 {dataset_name}: {e}")

    # 验证数据集注册
    print(f"\n数据集验证:")
    for dataset_name in registered_datasets:
        try:
            dataset_dicts = DatasetCatalog.get(dataset_name)
            sample_count = len(dataset_dicts)
            print(f"✓ {dataset_name}: {sample_count} 个样本")

            # 检查第一个样本的结构
            if sample_count > 0:
                first_sample = dataset_dicts[0]
                print(f"  - 示例图像: {first_sample.get('file_name', 'N/A')}")
                print(f"  - 注解数量: {len(first_sample.get('annotations', []))}")
        except Exception as e:
            print(f"✗ {dataset_name} 验证失败: {e}")

    return num_classes, thing_classes


def verify_dataset_structure():
    """验证数据集结构完整性"""
    print("\n=== 数据集结构验证 ===")

    # 检查注册的数据集
    registered_names = ["custom_train", "custom_val", "custom_test"]

    for name in registered_names:
        try:
            # 获取数据集
            dataset_dicts = DatasetCatalog.get(name)
            metadata = MetadataCatalog.get(name)

            print(f"\n{name}:")
            print(f"  样本数量: {len(dataset_dicts)}")
            print(f"  类别数量: {len(metadata.thing_classes)}")

            # 检查前几个样本
            for i, sample in enumerate(dataset_dicts[:3]):
                image_path = sample['file_name']
                annotations = sample.get('annotations', [])

                print(f"  样本 {i + 1}:")
                print(f"    图像: {os.path.basename(image_path)}")
                print(f"    图像存在: {os.path.exists(image_path)}")
                print(f"    注解数量: {len(annotations)}")

                if annotations:
                    categories = [ann.get('category_id', -1) for ann in annotations]
                    print(f"    类别ID: {list(set(categories))}")

        except Exception as e:
            print(f"  ✗ 验证失败: {e}")


def get_dataset_statistics():
    """获取数据集统计信息"""
    print("\n=== 数据集统计信息 ===")

    for dataset_name in ["custom_train", "custom_val", "custom_test"]:
        try:
            dataset_dicts = DatasetCatalog.get(dataset_name)
            metadata = MetadataCatalog.get(dataset_name)

            # 统计类别分布
            category_counts = {}
            total_instances = 0

            for sample in dataset_dicts:
                for ann in sample.get('annotations', []):
                    cat_id = ann.get('category_id', -1)
                    if 0 <= cat_id < len(metadata.thing_classes):
                        cat_name = metadata.thing_classes[cat_id]
                        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
                        total_instances += 1

            print(f"\n{dataset_name}:")
            print(f"  图像数量: {len(dataset_dicts)}")
            print(f"  实例总数: {total_instances}")
            print(f"  平均每图实例数: {total_instances / len(dataset_dicts):.2f}")

            print("  类别分布:")
            for cat_name, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_instances) * 100 if total_instances > 0 else 0
                print(f"    {cat_name}: {count} ({percentage:.1f}%)")

        except Exception as e:
            print(f"  ✗ {dataset_name} 统计失败: {e}")


if __name__ == "__main__":
    try:
        print("=== 开始注册数据集 ===")
        num_classes, thing_classes = register_custom_dataset()
        print(f"\n数据集注册完成！")
        print(f"类别数量: {num_classes}")
        print(f"类别列表: {thing_classes}")

        # 验证数据集结构
        verify_dataset_structure()

        # 获取统计信息
        get_dataset_statistics()

        print(f"\n=== 注册成功 ===")
        print("现在可以在训练脚本中使用以下数据集名称:")
        print("- custom_train")
        print("- custom_val")
        print("- custom_test")

    except Exception as e:
        print(f"注册失败: {e}")
        import traceback

        traceback.print_exc()