# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10jjedrlzEEImBdgXlAY_M8N63GhGmkv3
"""

# 📦 Установка окружения (если в Colab)
!pip install torch --quiet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# 🧠 Улучшенный TrustMetrixLayer V3
class TrustMetrixLayer(nn.Module):
    def __init__(self, input_dim, context_dim=4, name="TML"):
        super().__init__()
        self.name = name
        self.trust_weights = nn.Parameter(torch.ones(input_dim))  # базовое доверие
        self.context_gen = nn.Linear(input_dim, context_dim)       # обучаемый контекст
        self.context_weights = nn.Parameter(torch.randn(context_dim, input_dim) * 0.1)
        self.norm = nn.LayerNorm(input_dim)
        self.log = []

    def forward(self, x):
        context = self.context_gen(x)
        context_effect = torch.matmul(context, self.context_weights)
        modulator = torch.sigmoid(context_effect)  # контекст как маска доверия
        adjusted_weights = torch.clamp(self.trust_weights * modulator, 0.1, 2.0)
        weighted = x * adjusted_weights
        output = self.norm(weighted)

        # лог reasoning
        self.log.append({
            "input": x.detach().cpu().numpy(),
            "context": context.detach().cpu().numpy(),
            "weights": adjusted_weights.detach().cpu().numpy(),
            "output": output.detach().cpu().numpy()
        })
        return output

    def reflect(self, step=-1):
        if not self.log:
            return ["Нет reasoning."]
        record = self.log[step]
        return [
            f"{self.name} | Признак {i}: вход={i1:.3f}, контекст=({', '.join(f'{c:.2f}' for c in record['context'])}), "
            f"вес={w:.3f}, результат={o:.3f}"
            for i, (i1, w, o) in enumerate(zip(record["input"], record["weights"], record["output"]))
        ]

# 💾 Каскадная память reasoning
class CascadeMemory:
    def __init__(self, max_memory=1000):
        self.entries = []
        self.max_memory = max_memory

    def store(self, input_vector, context, trust_weights, prediction, true_label):
        self.entries.append({
            "input": input_vector,
            "context": context,
            "trust": trust_weights,
            "prediction": prediction,
            "true_label": true_label
        })
        if len(self.entries) > self.max_memory:
            self.entries.pop(0)

    def last_n(self, n=5):
        return self.entries[-n:]

# 🧱 Глубокая сеть с reasoning-логикой
class DeepCognitiveNet(nn.Module):
    def __init__(self, input_dim=10, context_dim=4, hidden_dims=[32]*9 + [16], output_dim=3):
        super().__init__()
        self.context_dim = context_dim
        self.trust_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.memory = CascadeMemory()

        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            self.trust_layers.append(TrustMetrixLayer(dims[i], context_dim, f"Trust-{i+1}"))
            self.linear_layers.append(nn.Linear(dims[i], dims[i+1]))
        self.out = nn.Linear(dims[-1], output_dim)

    def forward(self, x, y_true=None):
        original_input = x.clone().detach()
        for trust, linear in zip(self.trust_layers, self.linear_layers):
            x = trust(x)
            x = F.relu(linear(x))
        out = self.out(x)

        if y_true is not None:
            weights_combined = [trust.trust_weights.detach().cpu().numpy() for trust in self.trust_layers]
            context_vectors = [trust.log[-1]['context'] for trust in self.trust_layers]
            self.memory.store(original_input.numpy(), context_vectors, weights_combined, out.detach().numpy(), y_true.item())

        return out

    def reflect_all(self):
        return {f"Trust-{i+1}": trust.reflect() for i, trust in enumerate(self.trust_layers)}

    def show_memory(self, last_n=3):
        logs = self.memory.last_n(last_n)
        for i, entry in enumerate(logs):
            print(f"\n📌 Пример {i+1}: true_label={entry['true_label']}, prediction={entry['prediction'].argmax()}")
            print(f"Вход: {entry['input']}")
            print(f"Контексты: {[list(np.round(c, 2)) for c in entry['context'][:2]]} ...")
            print(f"Первый слой доверия: {np.round(entry['trust'][0][:5], 3)} ...")

# 📊 Данные
def generate_data(n=300, dim=10):
    X = torch.randn(n, dim)
    y = torch.randint(0, 3, (n,))
    return X, y

# 🚀 Обучение
model = DeepCognitiveNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
X_train, y_train = generate_data(500)

for epoch in range(10):
    total_loss = 0
    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i].unsqueeze(0)
        pred = model(x, y_true=y)
        loss = loss_fn(pred.unsqueeze(0), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 🧠 Вывод reasoning и памяти
print("\n🧠 Последний reasoning:")
reflections = model.reflect_all()
for name, reasoning in list(reflections.items())[:2]:  # Покажем первые 2 слоя
    print(f"\n--- {name} ---")
    for line in reasoning:
        print(line)

print("\n🧠 Последние 3 воспоминания из каскадной памяти:")
model.show_memory(3)