import os
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# 你的数据集路径配置
DATASET_ROOT = "/media/ubuntu/DATA4T/dataset/LIS_paixu/RGB-dark"

# 检查路径是否存在
if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(f"数据集路径不存在: {DATASET_ROOT}")

ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'images')  # 训练和测试图像在同一目录
TEST_PATH = os.path.join(DATASET_ROOT, 'images')

# 修改标注文件路径
TRAIN_JSON = os.path.join(ANN_ROOT, 'lis_coco_JPG_train+1.json')
TEST_JSON = os.path.join(ANN_ROOT, 'lis_coco_JPG_test+1.json')


# 如果你有NC4K数据集，可以保留这部分，否则注释掉
# NC4K_ROOT = 'NC4K'
# NC4K_PATH = os.path.join(NC4K_ROOT, 'Imgs')
# NC4K_JSON = os.path.join(NC4K_ROOT, 'nc4k_test.json')

def register_all_cis():
    """注册所有CIS相关数据集"""

    # 注册训练集
    if os.path.exists(TRAIN_JSON) and os.path.exists(TRAIN_PATH):
        register_coco_instances("cis_train", {}, TRAIN_JSON, TRAIN_PATH)
        print(f"✓ 训练集注册成功: {TRAIN_JSON}")
    else:
        print(f"✗ 训练集路径不存在: {TRAIN_JSON} 或 {TRAIN_PATH}")

    # 注册测试集
    if os.path.exists(TEST_JSON) and os.path.exists(TEST_PATH):
        register_coco_instances("cis_test", {}, TEST_JSON, TEST_PATH)
        print(f"✓ 测试集注册成功: {TEST_JSON}")
    else:
        print(f"✗ 测试集路径不存在: {TEST_JSON} 或 {TEST_PATH}")

    # 从注解文件中自动获取类别信息
    import json
    try:
        with open(TRAIN_JSON, 'r') as f:
            coco_data = json.load(f)

        # 提取类别信息
        categories = coco_data.get('categories', [])
        thing_classes = [cat['name'] for cat in categories]

        # 设置元数据
        for dataset_name in ["cis_train", "cis_test"]:
            try:
                MetadataCatalog.get(dataset_name).thing_classes = thing_classes
                MetadataCatalog.get(dataset_name).evaluator_type = "coco"
            except:
                pass  # 如果数据集不存在就跳过

        print(f"类别数量: {len(thing_classes)}")
        print(f"类别名称: {thing_classes}")

    except Exception as e:
        print(f"警告: 无法读取类别信息: {e}")
        # 使用你提供的默认类别
        thing_classes = ["bicycle", "chair", "diningtable", "bottle", "motorbike", "car", "tvmonitor", "bus"]

        # 设置默认类别
        for dataset_name in ["cis_train", "cis_test"]:
            try:
                MetadataCatalog.get(dataset_name).thing_classes = thing_classes
                MetadataCatalog.get(dataset_name).evaluator_type = "coco"
            except:
                pass

        print(f"使用默认类别数量: {len(thing_classes)}")
        print(f"默认类别名称: {thing_classes}")

    # 如果你还要保留原来的COD10K数据集注册，可以取消注释下面的代码
    # register_cod10k_dataset()
    # register_nc4k_dataset()


def register_cod10k_dataset():
    """注册COD10K数据集（如果需要的话）"""
    # 这里是原始的COD10K数据集注册代码
    # 你可以根据需要保留或删除
    pass


def register_nc4k_dataset():
    """注册NC4K数据集（如果需要的话）"""
    # 这里是原始的NC4K数据集注册代码
    # 你可以根据需要保留或删除
    pass


# 在模块加载时自动注册数据集
if __name__ != "__main__":
    register_all_cis()