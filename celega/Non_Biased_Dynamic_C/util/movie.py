import os
import hashlib
from moviepy import ImageSequenceClip
from PIL import Image

def calculate_image_hash(image_path):
    with Image.open(image_path) as img:
        return hashlib.md5(img.tobytes()).hexdigest()

def get_all_images(image_folder):
    all_images = []

    for img in sorted(os.listdir(image_folder)):
        if img.endswith('.png'):
            img_path = os.path.join(image_folder, img)
            all_images.append(img_path)

    return all_images

def delete_all_images(image_folder):
    for img in sorted(os.listdir(image_folder)):
        if img.endswith('.png'):
            img_path = os.path.join(image_folder, img)
            os.remove(img_path)
            print(f"Deleted image: {img_path}")

def compile_images_to_video(image_folder, output_video_path, fps=1):
    # Remove duplicate images and get a list of unique image file paths
    unique_images = get_all_images(image_folder)

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