import os
from PIL import Image
from glob import glob

def resize_and_pad_images(input_dir, output_dir, size=(1024, 1024)):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all image files in the input directory with .png extension
    input_files = glob(os.path.join(input_dir, "*.jpg"))  # Use "*.png" to match your initial pattern
    
    for input_path in input_files:
        # Determine the output path
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        # Open the input image
        image = Image.open(input_path)
        
        # Calculate the target size to maintain the aspect ratio
        aspect_ratio = image.width / image.height
        if aspect_ratio > 1:
            # Image is wider than tall
            new_width = size[0]
            new_height = round(new_width / aspect_ratio)
        else:
            # Image is taller than wide
            new_height = size[1]
            new_width = round(new_height * aspect_ratio)
        
        # Resize the image to maintain aspect ratio
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a new image with a black background
        new_image = Image.new("RGB", size, "black")
        
        # Calculate the positioning to center the image
        x = (size[0] - new_width) // 2
        y = (size[1] - new_height) // 2
        
        # Paste the resized image onto the center of the black background
        new_image.paste(image, (x, y))
        
        # Save the output image
        new_image.save(output_path)

def crop_center_images(input_dir, output_dir, crop_size=(1024, 1024)):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List all image files in the input directory with .jpg extension
    input_files = glob(os.path.join(input_dir, "*.jpg"))  # Use "*.jpg" for JPEG images
    
    for input_path in input_files:
        # Determine the output path
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)

        # Open the input image
        image = Image.open(input_path)
        width, height = image.size

        # Calculate the crop area
        left = (width - crop_size[0])/2
        top = (height - crop_size[1])/2-30
        right = (width + crop_size[0])/2
        bottom = (height + crop_size[1])/2-30

        # Crop the center of the image
        cropped_image = image.crop((left, top, right, bottom))
        
        # Save the cropped image
        cropped_image.save(output_path)



# # Example usage: Adjust the paths as necessary
# input_directory = "../input/pose_images/dancing003/"
# output_directory = "../input/pose_images/resize_out/dancing003_1024/"
# resize_and_pad_images(input_directory, output_directory)
# print("finished")
# Example usage: Adjust the paths as necessary
input_directory = "../input/pose_images/dancing003/"
output_directory = "../output/pose_images/cropped/dancing_1024/"
crop_center_images(input_directory, output_directory)
print("Finished cropping images.")