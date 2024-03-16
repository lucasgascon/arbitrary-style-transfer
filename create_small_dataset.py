import os
import random
import shutil
import argparse


def create_small_dataset(source_dir, target_dir, n):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get a list of all files in the source directory
    all_files = os.listdir(source_dir)

    # Filter out any non-image files if necessary (assuming images have extensions like .jpg, .png, etc.)
    image_files = [file for file in all_files if file.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Randomly select 1000 images
    selected_images = random.sample(image_files, n)

    # Copy the selected images to the target directory
    for image in selected_images:
        shutil.copy(os.path.join(source_dir, image),
                    os.path.join(target_dir, image))

    print(f'Copied {len(selected_images)} images to {target_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str,
                        help='The directory containing the original images')
    parser.add_argument('--target_dir', type=str,
                        help='The directory to copy the images to')
    parser.add_argument('--n', type=int, default=1000,
                        help='The number of images to copy')

    args = parser.parse_args()
    create_small_dataset(args.source_dir, args.target_dir, args.n)
