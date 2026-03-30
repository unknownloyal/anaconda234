import numpy as np
import matplotlib.pyplot as plt
t = ["I","love","deep","learning"]
np.random.seed(0)
X = np.random.rand(4,4)
Q,K,V = X@np.random.rand(4,4), X@np.random.rand(4,4), X@np.random.rand(4,4)
S = Q@K.T/2
W = np.exp(S)/np.sum(np.exp(S),axis=1,keepdims=True)
O = W@V
print("Weights:\n",W)
print("\nOutput:\n",O)
plt.imshow(W)
plt.xticks(range(4),t)
plt.yticks(range(4),t)
for i in range(4):
    for j in range(4):
        plt.text(j,i,f"{W[i,j]:.2f}",ha='center',color='white')
plt.show()

2.
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim, batch_size, epochs = 100, 128, 30
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
loader = DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)
G = nn.Sequential(
    nn.Linear(z_dim,256), nn.LeakyReLU(0.2),
    nn.Linear(256,512), nn.LeakyReLU(0.2),
    nn.Linear(512,1024), nn.LeakyReLU(0.2),
    nn.Linear(1024,784), nn.Tanh()
).to(device)
D = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,512), nn.LeakyReLU(0.2),
    nn.Linear(512,256), nn.LeakyReLU(0.2),
    nn.Linear(256,1), nn.Sigmoid()
).to(device)
loss = nn.BCELoss()
optG = optim.Adam(G.parameters(),0.0002,(0.5,0.999))
optD = optim.Adam(D.parameters(),0.0002,(0.5,0.999))
for epoch in range(1,epochs+1):
    for real,_ in loader:
        real = real.to(device)
        bs = real.size(0)
        real_label = torch.ones(bs,1).to(device)
        fake_label = torch.zeros(bs,1).to(device)
        z = torch.randn(bs,z_dim).to(device)
        fake = G(z).view(-1,1,28,28)
        lossD = loss(D(real),real_label) + loss(D(fake.detach()),fake_label)
        optD.zero_grad(); lossD.backward(); optD.step()
        z = torch.randn(bs,z_dim).to(device)
        fake = G(z).view(-1,1,28,28)
        lossG = loss(D(fake),real_label)
        optG.zero_grad(); lossG.backward(); optG.step()
    print(f"Epoch {epoch}/{epochs} Loss_D:{lossD.item():.4f} Loss_G:{lossG.item():.4f}")
    if epoch % 5 == 0:
        with torch.no_grad():
            z = torch.randn(16,z_dim).to(device)
            samples = G(z).view(-1,1,28,28).cpu()*0.5 + 0.5
        fig,ax = plt.subplots(1,16,figsize=(16,2))
        for i in range(16):
            ax[i].imshow(samples[i][0],cmap="gray")
            ax[i].axis("off")
        plt.title(f"Epoch {epoch}")
        plt.show()

 

3.
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
texts = ["I love this movie","This is terrible"]
labels = [1,0]
name = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
def pred(m, t):
    x = tok(t, return_tensors="pt")
    return m(**x).logits.softmax(-1)
# Before
print("Before:")
print(pred(model,"I love this movie"))
print(pred(model,"This is terrible"))
# Apply LoRA
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin","v_lin"])
model = get_peft_model(model, cfg)
# Train (simple)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
for _ in range(3):
    for t,l in zip(texts,labels):
        out = model(**tok(t,return_tensors="pt"), labels=torch.tensor([l]))
        opt.zero_grad(); out.loss.backward(); opt.step()
# After
print("\nAfter:")
print(pred(model,"I love this movie"))
print(pred(model,"This is terrible"))

4.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt",truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
question = "If 1 pen costs 10 rupees, what is the cost of 5 pens?"
zero_prompt = f"""
Answer the question:
{question}
"""
zero_output = generate_text(zero_prompt)
few_prompt = f"""
Solve the following:
Q: If 1 pen costs 10 rupees, what is the cost of 3 pens?
A: 3 * 10 = 30 rupees
Q: If 1 pencil costs 10 rupees, what is the cost of 4 pencils?
A: 4 * 10 = 40 rupees
Q: If 1 book costs 10 rupees , what is the cost of 5 books?
A: 5 * 10 = 50 rupees
Q: {question}
A:
"""
few_output = generate_text(few_prompt)
cot_prompt = f"""
Let's slove step by step and then give final answer:
Q: {question}
Final Answer:
"""
cot_output = generate_text(cot_prompt)

print("===== ZERO-SHOT OUTPUT =====")
print(zero_output)
print("\n===== FEW-SHOT OUTPUT =====")
print(few_output)
print("\n===== CHAIN-OF-THOUGHT OUTPUT =====")
print(cot_output)

5.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
texts = ["I love this movie", "Amazing film",
         "I hate this movie", "Terrible film"]
labels = [1, 1, 0, 0]   # 1 = positive, 0 = negative
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
# Discriminative
lr = LogisticRegression()
lr.fit(X, labels)
nb = MultinomialNB()
nb.fit(X, labels)
test = ["I love this", "Bad movie"]
X_test = vectorizer.transform(test)
print("Logistic Regression (Discriminative):")
print(lr.predict(X_test))
print("\nNaive Bayes (Generative):")
print(nb.predict(X_test))
