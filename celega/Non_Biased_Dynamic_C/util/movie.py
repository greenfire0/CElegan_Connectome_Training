import os
import hashlib
from moviepy.editor import ImageSequenceClip
from PIL import Image

def calculate_image_hash(image_path):
    with Image.open(image_path) as img:
        return hashlib.md5(img.tobytes()).hexdigest()

def remove_duplicate_images(image_folder):
    unique_hashes = set()
    unique_images = []

    for img in sorted(os.listdir(image_folder)):
        if img.endswith('.png'):
            img_path = os.path.join(image_folder, img)
            img_hash = calculate_image_hash(img_path)

            if img_hash not in unique_hashes:
                unique_hashes.add(img_hash)
                unique_images.append(img_path)
            else:
                os.remove(img_path)
                print(f"Removed duplicate image: {img_path}")

    return unique_images

def delete_all_images(image_folder):
    for img in sorted(os.listdir(image_folder)):
        if img.endswith('.png'):
            img_path = os.path.join(image_folder, img)
            os.remove(img_path)
            print(f"Deleted image: {img_path}")

def compile_images_to_video(image_folder, output_video_path, fps=1):
    # Remove duplicate images and get a list of unique image file paths
    unique_images = remove_duplicate_images(image_folder)

    # Create a video clip from the unique image sequence
    clip = ImageSequenceClip(unique_images, fps=fps)
    
    # Write the video file to the specified output path
    clip.write_videofile(output_video_path, codec='libx264')
    
    # Delete all images in the folder after video generation
    delete_all_images(image_folder)

# Specify the folder containing the images and the output video file path
if __name__=="main":
    image_folder = '/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/tmp_img'
    output_video_path = 'weight_matrix_video_unclipped_patternfood.mp4'

    # Compile the images into a video
    compile_images_to_video(image_folder, output_video_path, fps=3)