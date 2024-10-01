import os
import subprocess

# Specify the path to the folder containing the image folders
input_folder = "/scratch_xijun/data/VLEP/videos/vlep_frames"
output_folder = "/scratch_xijun/data/VLEP/videos/vlep_videos"

# List all subdirectories (image folders) in the input folder
subdirectories = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

print('video number:', len(subdirectories))
count = 0
# Loop through each subdirectory
for directory in subdirectories:
    input_folder_path = os.path.join(input_folder, directory)
    output_video_path = os.path.join(output_folder, f"{directory}.mp4")

    # Use ffmpeg to convert images to video
    subprocess.run(['ffmpeg', '-framerate', '3', '-pattern_type', 'glob', '-i', f'{input_folder_path}/*.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video_path])

    #print(f"Video created for folder: {directory}")
    count +=1

print('successed video number:', count)
