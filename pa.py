pip install matplotlib diffusers transformers torch
 

1.
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe=pipe.to("cpu")
prompt = "A futuristic city with flying cars."
image = pipe(prompt).images[0]
image.save("output1.png")
image.show()
2.
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
prompt="A medical image of human heart detailed scan"
image = pipe(prompt).images[0]
image.save("output2.png")
image.show()
or
from diffusers import StableDiffusionPipeline
import torch
device="cpu"
b_m="runwayml/stable-diffusion-v1-5"
pipe=StableDiffusionPipeline.from_pretrained(b_m,torch_dtype=torch.float32).to(device)
l_m="latent-consistency/lcm-lora-sdv1-5"
pipe.load_lora_weights(l_m)
pipe.fuse_lora()
prompt="A diseased tomato leaf with brown spots, agricultural crop disease, high detail"
image=pipe(prompt,num_inference_steps =10,guidance_scale = 1.0).images[0]
image.show()

3.
from transformers import pipeline
qa=pipeline("text-generation",model="gpt2")
questions=[
"what is Artificial Intelligence?",
"what is Machine Learning?"
]
for q in questions:
result=qa(q,max_length=100)
print(q)
print(result[0]["generated_text"])
 

4.
from transformers import pipeline
generator=pipeline('text-generation',model="gpt2")
text = """
Artificial Intelligence is transforming various industries by enabling automation,
improving decision-making, and enhancing user experiences. It is widely used in
healthcare, education, finance, and transportation. AI systems can analyze large
amounts of data quickly and accurately. However, there are challenges such as
data privacy, ethical concerns, and job displacement.
"""
prompt="Summarize the text:\n"+text+"\nSummary"
result=generator(prompt,max_new_tokens=40,do_sample=False)
output=result[0]['generated_text'].split("summary:")[-1]
print(output.strip())
 

5.
from transformers import pipeline
generator=pipeline("text-generation",model="gpt2")
prompt="AI is changing education by"
result=generator(prompt,max_length=50)
output=result[0]["generated_text"]
print(output)
 

6.
from transformers import pipeline
generator=pipeline("text-generation",model="gpt2")
prompt="""
write about applications of generative AI in healthcare.
1.Diagnosis
2.Drug discovery
3.Patient care
"""
result=generator(prompt,max_length=50)
output=result[0]["generated_text"]
print(output)
