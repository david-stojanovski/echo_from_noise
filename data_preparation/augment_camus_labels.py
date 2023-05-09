import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from natsort import natsorted
from torchio import RandomElasticDeformation


def postprocess_tensor(in_tensor):
    transform = transforms.ToPILImage()
    img_out = transform(in_tensor[0])
    return img_out


def augment_img_tensor(label_tensor):
    affine_degrees = (-5, 5)
    affine_translate = (0, 0.05)
    affine_scale = (0.8, 1.05)
    affine_shear = 5

    elastic_num_control_points = (10, 10, 4)
    elastic_locked_borders = 1
    elastic_max_displacement = (0, 30, 30)
    elastic_image_interpolation = 'nearest'

    img_transforms = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=affine_degrees,
                translate=affine_translate,
                scale=affine_scale,
                shear=affine_shear,
            ),
            RandomElasticDeformation(
                num_control_points=elastic_num_control_points,
                locked_borders=elastic_locked_borders,
                max_displacement=elastic_max_displacement,
                image_interpolation=elastic_image_interpolation
            )
        ])

    img_out = img_transforms(label_tensor)

    img_out = postprocess_tensor(img_out)
    return np.array(img_out)


def add_labels2camus(in_label_img, in_real_img, augment_img=False, plot_imgs=False):
    in_real_img = np.asarray(in_real_img)
    out_label_img = np.copy(in_label_img)
    cone_label = np.zeros_like(in_label_img)
    cone_label[in_real_img != 0] = 1  # create cone mask from real image

    if augment_img:
        out_label_img = augment_img_tensor(transforms.functional.to_tensor(out_label_img).unsqueeze(0))

    unclipped_labels = np.copy(out_label_img)
    out_label_img *= cone_label  # clips labels that are outside the cone
    out_label_img[out_label_img > 1] += 1  # make space for new label to match already generated datasets
    out_label_img[(out_label_img == 0) & (cone_label == 1)] = 2

    if plot_imgs:
        f, axarr = plt.subplots(1, 3)

        axarr[0].imshow(in_label_img, cmap='gray', vmin=0, vmax=np.max(in_label_img))
        axarr[0].title.set_text('Original')
        axarr[1].imshow(out_label_img, cmap='gray', vmin=0, vmax=out_label_img.max())
        axarr[1].title.set_text('augmented sector annotation')
        axarr[2].imshow(unclipped_labels, cmap='gray', vmin=0, vmax=unclipped_labels.max())
        axarr[2].title.set_text('augmented annotation')
        plt.show()

    return out_label_img, unclipped_labels.astype(np.uint8)


def generate_save_path(view_folder, folder_type, train_or_val_type, patient_id, img_count):
    save_path = os.path.join(view_folder, folder_type, train_or_val_type, patient_id + '_' + str(img_count) + '.png')
    os.makedirs(os.path.join(save_view_folder, folder_type, train_or_val), exist_ok=True)
    return save_path


if __name__ == '__main__':
    data_folder = r'/path/to/camus/data'
    save_folder = r'/path/to/save'
    n_repeat = 5
    add_augmentation = True
    debug_plot = False

    view_folders = ['2CH_ES', '2CH_ED', '4CH_ES', '4CH_ED']
    for view_folder in view_folders:
        root_folder = os.path.join(data_folder, view_folder)
        save_view_folder = os.path.join(save_folder, view_folder + '_augmented')

        all_label_paths = natsorted(glob.glob(os.path.join(root_folder, 'annotations', '*', '*.png')))

        for label_path in all_label_paths:
            patient = os.path.splitext(label_path.split(os.sep)[-1])[0]
            train_or_val = label_path.split(os.sep)[-2]

            label_path = glob.glob(os.path.join(root_folder, 'annotations', train_or_val, patient + '*'))[0]
            real_img_path = glob.glob(os.path.join(root_folder, 'images', train_or_val, patient + '*'))[0]

            label_img = Image.open(label_path)
            real_img = Image.open(real_img_path)

            for ii in range(n_repeat):
                label_save_path = generate_save_path(save_view_folder, 'annotations', train_or_val, patient, ii)
                img_save_path = generate_save_path(save_view_folder, 'images', train_or_val, patient, ii)

                if train_or_val == 'testing':
                    label_img.save(label_save_path)
                    shutil.copy2(real_img_path, img_save_path)
                else:
                    sector_labelmap, labelmap = add_labels2camus(label_img, real_img, augment_img=add_augmentation,
                                                                 plot_imgs=debug_plot)

                    sector_label_save_path = generate_save_path(save_view_folder, 'sector_annotations', train_or_val,
                                                                patient, ii)

                    Image.fromarray(labelmap).save(label_save_path)
                    Image.fromarray(sector_labelmap).save(sector_label_save_path)
                    shutil.copy2(real_img_path, img_save_path)

            print('Saved images for: ', label_path)
