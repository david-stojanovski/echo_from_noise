import glob
import os

import SimpleITK as sitk
import cv2
import numpy as np
from natsort import natsorted


def subsample_segmentation(in_data, dim_size=None):
    """Takes an input image (no color channel) and resamples it to a desired size. It can be expanded to work with
    any number of dimensions. """
    dims = np.shape(in_data)
    x_val, y_val = np.mgrid[0:dims[0] - 1:complex(0, dim_size[0]),
                   :dims[1] - 1:complex(0, dim_size[1])].astype(int)
    return in_data[x_val, y_val]


def load_mhd_img(mhd_path, img_out_size=None):
    itk_image = sitk.ReadImage(mhd_path)
    image_array = np.squeeze(sitk.GetArrayViewFromImage(itk_image))
    return subsample_segmentation(image_array, dim_size=img_out_size)


def generate_save_path(save_folder_path, folder_type, save_patient_name, save_selected_view_name, save_train_test_type):
    images_save_folder = os.path.join(save_folder_path, folder_type, save_train_test_type)
    os.makedirs(images_save_folder, exist_ok=True)
    out_save_path = os.path.join(images_save_folder,
                                 '_'.join([save_patient_name, save_selected_view_name, save_train_test_type]) + '.png')
    return out_save_path


def save_all_camus_imgs(in_data_folder, save_folder, img_save_size, selected_view_name,
                        train_test_type):
    patient_folders = natsorted(glob.glob(os.path.join(in_data_folder, 'patient*')))
    for idx, patient_folder in enumerate(patient_folders):
        __, patient_name = os.path.split(patient_folder)
        patient_idx = int(patient_name.split('patient')[-1])

        patient_img_file_name = patient_name + '_' + selected_view_name + '.mhd'
        patient_gt_file_name = patient_name + '_' + selected_view_name + '_gt.mhd'

        loaded_img = load_mhd_img(os.path.join(patient_folder, patient_img_file_name), img_out_size=img_save_size)
        loaded_gt_img = load_mhd_img(os.path.join(patient_folder, patient_gt_file_name), img_out_size=img_save_size)

        if patient_idx <= 50 and train_test_type == 'training':  # first 50 patients in training are set to validation
            save_train_test_type = 'validation'
        else:
            save_train_test_type = train_test_type

        new_view_folder = os.path.join(save_folder, selected_view_name)
        all_views_folder = os.path.join(save_folder, 'all_views')

        img_save_path = generate_save_path(new_view_folder, 'images', patient_name, selected_view_name,
                                           save_train_test_type)
        annotations_save_path = generate_save_path(new_view_folder, 'annotations', patient_name, selected_view_name,
                                                   save_train_test_type)
        all_views_img_save_path = generate_save_path(all_views_folder, 'images', patient_name, selected_view_name,
                                                     save_train_test_type)
        all_views_annotations_save_path = generate_save_path(all_views_folder, 'annotations', patient_name,
                                                             selected_view_name, save_train_test_type)

        cv2.imwrite(img_save_path, loaded_img)
        cv2.imwrite(annotations_save_path, loaded_gt_img)
        cv2.imwrite(all_views_img_save_path, loaded_img)
        cv2.imwrite(all_views_annotations_save_path, loaded_gt_img)

    print('Done with {} patients for view {} {}'.format(idx + 1, selected_view_name, save_train_test_type))


if __name__ == '__main__':

    camus_data_folder = r'/path/to/camus/data'
    save_folder_path = r'/path/to/save/folder'
    save_img_size = (256, 256)

    view_names = ['2CH_ED', '2CH_ES', '4CH_ED', '4CH_ES']
    train_test_names = ['training', 'testing']
    for view_name in view_names:
        for name in train_test_names:
            data_folder = os.path.join(camus_data_folder, name)

            save_all_camus_imgs(data_folder, save_folder_path, save_img_size, selected_view_name=view_name,
                                train_test_type=name)
