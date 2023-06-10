import os
from tqdm import tqdm


def sort(img_list):
    img_list.sort()
    img_list.sort(key=lambda x: int(x[:-4]))
    return img_list


def data_generate(ffhq_train_path, sys_train_path, test_path, data_txt_save_path):
    train_txt = os.path.join(data_txt_save_path, "train_img.txt")
    val_txt = os.path.join(data_txt_save_path, "val_img.txt")
    pbar_ffhq = tqdm(total=len(os.listdir(ffhq_train_path)))
    pbar_sys = tqdm(total=len(os.listdir(sys_train_path)))
    pbar_test = tqdm(total=len(os.listdir(test_path)))
    ffhq_id_list = os.listdir(ffhq_train_path)
    ffhq_id_list.sort()
    # ffhq_id_list.remove('check_for_dataset.py')
    # ffhq_id_list.remove('LICENSE.txt')
    with open(train_txt, 'a') as train_txt_img:
        for i, image_name in enumerate(sort(os.listdir(sys_train_path))):
            image_path = os.path.join(sys_train_path, image_name)

            train_txt_img.write(image_path)
            train_txt_img.write('\n')
            pbar_sys.update()
        # for i, image_id in enumerate(ffhq_id_list):
        #     id_file = os.path.join(ffhq_train_path, image_id)
        #     for k, image_name in enumerate(sort(os.listdir(id_file))):
        #         image_path = os.path.join(id_file, image_name)
        #         train_txt_img.write(image_path)
        #         train_txt_img.write('\n')
        #     pbar_ffhq.update()

        for i, image_name in enumerate(ffhq_id_list):
            image_path = os.path.join(ffhq_train_path, image_name)
            train_txt_img.write(image_path)
            train_txt_img.write('\n')
            pbar_ffhq.update()
    train_txt_img.close()
    with open(val_txt, 'a') as val_txt_img:
        for i, image_name in enumerate(sort(os.listdir(test_path))):
            image_path = os.path.join(test_path, image_name)
            val_txt_img.write(image_path)
            val_txt_img.write('\n')
            pbar_test.update()
    val_txt_img.close()


if __name__ == '__main__':
    ffhq_path = '/apdcephfs_cq2/share_1290939/liuhongyu/data/cars_train'
    sys_path = '/apdcephfs/share_1290939/kumamzqliu/data/car_inversion/train/image'
    val_path = '/apdcephfs/share_1290939/kumamzqliu/data/car_inversion/test/image'
    data_save = '/apdcephfs/share_1290939/kumamzqliu/data/car_inversion'
    data_generate(ffhq_path, sys_path, val_path, data_save)
