import torch
import imageio
import os 
from diffusers import AutoPipelineForText2Image,ControlNetModel
from diffusers.utils import load_image
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
from datetime import datetime


print(os.getcwd())
model_path = "../checkpoint/"+"majicmixRealistic_v7.safetensors"
# model_path = "../checkpoint/"+"stable-diffusion-xl-base-1.0"

controlnet_path="../checkpoint/"+"control_v11p_sd15_openpose"
# controlnet_path="../checkpoint/"+"controlnet-openpose-sdxl-1.0"

lora_name="Zendaya"+".safetensors"
lora_path="../checkpoint/loras/"

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


pose_name = "dancing_512_01"
pose_path = '../input/pose_images/'+pose_name
pose_images = []
filenames = sorted([f for f in os.listdir(pose_path) if f.endswith((".jpg", ".png"))])  # Add other file types if needed

for filename in filenames:
    img_path = os.path.join(pose_path, filename)
    pose_images.append(load_image(img_path))
frames_num=len(pose_images)

lora_scale=0.8
repeated_times =10
num_inference_steps=80




prompts = ["relastic,extremely detailed,woman,sophisticated,8k uhd,short hair,dancing,beach,white jacket,brown eyes,sunset","relastic,woman, dancing, house, black jacket, dress,long hair"]

negative_prompt ="(bad-hands-5), interlocked fingers, poorly drawn fingers, (easynegative), (worst quality, low quality:2), twins, closed eyes, missing teeth, extra arms, blurry face, fat belly, heterochromia, big ears, bald, brunette, dull hair, furry, zombie, granny, gore, (watermark)"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)



#generator = torch.Generator("cuda").manual_seed(31)

pipeline = AutoPipelineForText2Image.from_pretrained(model_path,controlnet = controlnet,torch_dtype=torch.float16,requires_safety_checker=False,safety_checker=None).to("cuda")
# pipeline.load_lora_weights(lora_path, weight_name=lora_name)


pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipeline.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))


for k in range(repeated_times):
    latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(frames_num, 1, 1, 1)
    for i in range(len(prompts)):
        image = pipeline(prompt = [prompts[i]]*frames_num,negative_prompt=[negative_prompt]*frames_num,image = pose_images,num_inference_steps = num_inference_steps,latents = latents).images
        # lora
        # image = pipeline(prompt = [prompts[i]]*frames_num,negative_prompt=[negative_prompt]*frames_num,image = pose_images,lora_scale = lora_scale,num_inference_steps = num_inference_steps,latents = latents).images
        for j in range (frames_num):
            file_path = "../output/{}/test_pose_video/v1.5_512/{}_{}.png".format(datetime.now().date(), pose_name,datetime.now().time())
            ensure_dir(file_path)
            image[j].save(file_path)


pose_name = "dancing_512_02"
pose_path = '../input/pose_images/'+pose_name
pose_images_2 = []
filenames = sorted([f for f in os.listdir(pose_path) if f.endswith((".jpg", ".png"))])  # Add other file types if needed
for filename in filenames:
    img_path = os.path.join(pose_path, filename)
    pose_images_2.append(load_image(img_path))
frames_num_2=len(pose_images_2)


for k in range(repeated_times):
    latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(frames_num_2, 1, 1, 1)
    for i in range(len(prompts)):
        image = pipeline(prompt = [prompts[i]]*frames_num_2,negative_prompt=[negative_prompt]*frames_num_2,image = pose_images_2,num_inference_steps = num_inference_steps,latents = latents).images
        # lora
        # image = pipeline(prompt = [prompts[i]]*frames_num_2,negative_prompt=[negative_prompt]*frames_num_2,image = pose_images_2,lora_scale = lora_scale,num_inference_steps = num_inference_steps,latents = latents).images
        for j in range (frames_num_2):
            file_path = "../output/{}/test_pose_video/v1.5_512/{}_{}.png".format(datetime.now().date(), pose_name,datetime.now().time())
            # file_path = "../output/{}/test_pose_video/v1.5_512/{}_{}_{:02d}_{:03d}_{:02d}.png".format(datetime.now().date(), pose_name,datetime.now().time(), k, i+1, j)
            ensure_dir(file_path)
            image[j].save(file_path)



