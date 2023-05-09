import glob
import os
import pathlib
import shutil


def move_test_files_to_folder(data_folder_in, view_name_in, save_folder_testing):
    testing_img_data_folder = os.path.join(data_folder_in, 'images', 'testing')
    testing_label_data_folder = os.path.join(data_folder_in, 'annotations', 'testing')

    testing_img_save_folder = os.path.join(save_folder_testing, view_name_in, 'images', 'testing')
    testing_label_save_folder = os.path.join(save_folder_testing, view_name_in, 'annotations', 'testing')

    shutil.copytree(testing_img_data_folder, testing_img_save_folder, dirs_exist_ok=True)
    shutil.copytree(testing_label_data_folder, testing_label_save_folder, dirs_exist_ok=True)


if __name__ == '__main__':

    testing_data_folder = r'/path/to/camus/images'
    sdm_results_folder = r'/path/to/sdm/results'
    save_folder = r'/path/to/save/folder'

    samples_folder = os.path.join(sdm_results_folder, 'samples')
    labels_folder = os.path.join(sdm_results_folder, 'labels')

    all_inferenced_images = glob.glob(os.path.join(samples_folder, '*.png'))
    print('Found {} inferenced images'.format(len(all_inferenced_images)))

    all_inferenced_labels = glob.glob(os.path.join(labels_folder, '*.png'))
    print('Found {} inferenced labels'.format(len(all_inferenced_labels)))

    if len(all_inferenced_images) != len(all_inferenced_labels):
        raise ValueError('Number of inferenced images and labels do not match')

    for idx, inf_image_path in enumerate(all_inferenced_images):
        inf_img_name = inf_image_path.split(os.sep)[-1]
        patient_id = inf_img_name.split('_')[0]
        view_name = '_'.join(inf_img_name.split('_')[1:3])
        train_or_val = inf_img_name.split('_')[3]

        save_img_path = os.path.join(save_folder, view_name, 'images', train_or_val, inf_img_name)
        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
        save_label_path = os.path.join(save_folder, view_name, 'annotations', train_or_val, inf_img_name)
        os.makedirs(os.path.dirname(save_label_path), exist_ok=True)

        shutil.copy2(inf_image_path, save_img_path)
        label_file_path = pathlib.Path(str(inf_image_path).replace('samples', "labels"))
        shutil.copy2(label_file_path, save_label_path)

    move_test_files_to_folder(testing_data_folder, view_name, save_folder)
