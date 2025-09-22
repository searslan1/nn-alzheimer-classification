import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Ayarlar
RAW_DIR = "../data/raw/train"
PROC_DIR = "../data/processed"
VAL_RATIO = 0.2
SEED = 42

random.seed(SEED)

def prepare_data():
    # processed klasör yapısını kur
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(PROC_DIR, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)

    # train klasöründen val ayır
    class_names = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    for cls in class_names:
        src_cls_dir = os.path.join(RAW_DIR, cls)
        images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]

        train_imgs, val_imgs = train_test_split(images, test_size=VAL_RATIO, random_state=SEED)

        # klasörleri oluştur
        for split, split_imgs in zip(["train","val"], [train_imgs, val_imgs]):
            dst_cls_dir = os.path.join(PROC_DIR, split, cls)
            os.makedirs(dst_cls_dir, exist_ok=True)
            for fname in split_imgs:
                shutil.copy(os.path.join(src_cls_dir, fname), os.path.join(dst_cls_dir, fname))

    # test setini doğrudan kopyala
    raw_test = "data/raw/test"
    if os.path.exists(raw_test):
        for cls in os.listdir(raw_test):
            src_cls_dir = os.path.join(raw_test, cls)
            dst_cls_dir = os.path.join(PROC_DIR, "test", cls)
            os.makedirs(dst_cls_dir, exist_ok=True)
            for fname in os.listdir(src_cls_dir):
                if fname.lower().endswith((".jpg",".png",".jpeg")):
                    shutil.copy(os.path.join(src_cls_dir, fname), os.path.join(dst_cls_dir, fname))

    print("✅ Veri hazırlandı. data/processed/ içinde train, val ve test klasörleri oluşturuldu.")

if __name__ == "__main__":
    prepare_data()
