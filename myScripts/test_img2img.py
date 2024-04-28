import torch
import imageio
import os 
from diffusers import AutoPipelineForText2Image,ControlNetModel,AutoPipelineForImage2Image
from diffusers.utils import load_image
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

print(os.getcwd())
model_path = "../checkpoint/"+"stable-diffusion-v1-5"
model_path = "../checkpoint/"+"stable-diffusion-xl-base-1.0"


controlnet_path="../checkpoint/"+"control_v11p_sd15_openpose"
controlnet_path="../checkpoint/"+"controlnet-openpose-sdxl-1.0"


lora_name="Zendaya"+".safetensors"
lora_path="../checkpoint/loras/"
lora_scale=0.8

pose_path = '../input/pose_images/dancing_20'
pose_name = "dancing"
#pose_path = '../input/pose_images/'+pose_name


repeated_times =3
num_inference_steps=70

pose_images = []
filenames = sorted([f for f in os.listdir(pose_path) if f.endswith((".jpg", ".png"))])  # Add other file types if needed

for filename in filenames:
    img_path = os.path.join(pose_path, filename)
    pose_images.append(load_image(img_path))


prompts = ["extremely detailed,woman,sophisticated,8k uhd,short hair,dancing,beach,white jacket,brown eyes,sunset"]

negative_prompt ="(bad-hands-5), interlocked fingers, poorly drawn fingers, missing fingers, (easynegative), (worst quality, low quality:2), twins, closed eyes, missing teeth, extra arms, blurry face, fat belly, heterochromia, big ears, bald, brunette, dull hair, furry, zombie, granny, gore, (watermark)"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

frames_num=len(pose_images)

#generator = torch.Generator("cuda").manual_seed(31)

pipeline = AutoPipelineForImage2Image.from_pretrained(model_path,controlnet = controlnet,torch_dtype=torch.float16,requires_safety_checker=False,safety_checker=None).to("cuda")
pipeline.load_lora_weights(lora_path, weight_name=lora_name)

pipeline.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipeline.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))



# fix latents for all frames
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(frames_num, 1, 1, 1)

for k in range(repeated_times):
    for i in range(len(prompts)):
        for j in range (frames_num):
            image = pipeline(prompt = [prompts[i]]*frames_num,negative_prompt=[negative_prompt]*frames_num,image = pose_images,lora_scale = lora_scale,num_inference_steps = num_inference_steps,target_size=(1024, 1024),latents = latents).images
            image[j].save("../output/test_video/{}_{}_{:02d}_{:03d}_{:02d}.png".format(lora_name,pose_name,k,i+1,j))

#,latents = latents
# for k in range(repeated_times):
#     for i in range(len(prompts)):
#         for j in range (frames_num):
#             image = pipeline(prompt = [prompts[i]]*frames_num,negative_prompt=[negative_prompt]*frames_num,image = pose_images,lora_scale = lora_scale,latents = latents,num_inference_steps = num_inference_steps,negative_original_size=(512, 512),negative_target_size=(1024, 1024),).images
#             image[j].save("../output/test_video/{}_{}_{}{:03d}_{:02d}.png".format(lora_name,pose_name,k,i+1,j))
# lora_name="Zendaya"+".safetensors"

# pipeline.load_lora_weights(lora_path, weight_name=lora_name)






prompts = ["extremely detailed photo of a woman,sophisticated, 8k uhd,women,short hair,dancing in forest ,white jacket,sunset"]

# pipeline = AutoPipelineForText2Image.from_pretrained(model_path,torch_dtype=torch.float16,requires_safety_checker=False,safety_checker=None).to("cuda")
# pipeline.load_lora_weights(lora_path, weight_name=lora_name)
# image = pipeline([prompts[0]]*frames_num,negative_prompt=[negative_prompt]*frames_num,lora_scale = lora_scale,num_inference_steps = num_inference_steps).images
# image[0].save("../output/test_video/{}{:03d}_{:02d}_{}.png".format(lora_name,2,0,0))

# image = pipeline([prompts[0]]*frames_num,negative_prompt=[negative_prompt]*frames_num,image = pose_images,lora_scale = lora_scale,num_inference_steps = num_inference_steps).images
# image[0].save("../output/test_video/{}{:03d}_{:02d}_{}.png".format(lora_name,2,0,0))


# image[0].save("../output/test_video/{}{:03d}_{:02d}_{}.png".format(lora_name,2,0,0))

