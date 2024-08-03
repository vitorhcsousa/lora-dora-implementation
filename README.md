# LoRA and DoRA Implementation

This repository contains the implementation of LoRA and DoRA layers as proposed in the following papers:
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)

These layers are used in a Multi-Layer Perceptron (MLP) model.

## 🔍 LoRA and DoRA Layers

### 🔺 LoRA (Low-Rank Adaptation)

LoRA is designed to reduce computational costs and memory usage during fine-tuning of large pre-trained models. By updating only a subset of parameters using low-rank matrices, LoRA allows efficient adaptation to specific tasks, especially when computational resources are limited.
#### 🔑 Key Concepts

1. **Low-Rank Matrices**: In LoRA, two low-rank matrices, $A$ and $B$, are introduced. These matrices have a much smaller number of parameters compared to the original weight matrix $W$. During fine-tuning, instead of updating the full weight matrix, only these low-rank matrices are updated.

2. **Weight Update**: The weight update in LoRA can be represented as:

   
$$
W' = W + \alpha \cdot A \cdot B
$$

Here, $W$ is the original weight matrix, $A$ and $B$ are low-rank matrices, and $\alpha$ is a scaling factor that controls the impact of the adaptation. The product $A \cdot B$ approximates the change required in the weight matrix, and $\alpha$ scales this change.

3. **Dimensionality Reduction**: By using low-rank matrices, LoRA captures essential adaptations in a lower-dimensional subspace, reducing the number of learnable parameters and enhancing training efficiency.

4. **Efficiency**: The reduced number of parameters in $A$ and $B$ speeds up training and mitigates overfitting by limiting the number of parameters.

5. **Applications**: LoRA is beneficial in transfer learning, where a pre-trained model needs quick adaptation to new tasks with limited data.

### 🧭 DoRA (DoRA: Weight-Decomposed Low-Rank Adaptation)

DoRA extends the concept of LoRA by decomposing the pretrained weight matrix into a magnitude vector and a directional matrix. This allows the model to adapt more flexibly to new tasks by dynamically adjusting the low-rank matrices based on the current state of the training process, providing improved adaptability and efficiency.
#### Mathematical Explanation

In DoRA, the weight update is represented as:


$$
W' = m \frac{V + \Delta V}{\|V + \Delta V\|_c} = m \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$


where:
- $W'$ is the updated weight matrix.
- $m$ is the learned magnitude vector.
- $V$ is the initial directional matrix.
- $\Delta V$ represents the update to the directional component matrix $V$.
- $W_0$ is the initial pretrained weight matrix.
- $BA$ is the low-rank update applied to $W_0$.
- $\| \cdot \|_c$ denotes the vector-wise norm used for normalization.

#### Magnitude Vector and Directional Matrix

The magnitude vector $m$ and the directional matrix $V$ are used to dynamically adjust the low-rank matrices. The magnitude vector $m$ is defined as:

$$
m = \frac{\lVert W \rVert}{\lVert V \rVert}
$$

where $\lVert W \rVert$ is the norm of the original weight matrix $W$ and $\lVert V \rVert$ is the norm of the directional matrix $V$.

The magnitude vector $m$ scales the updates to the low-rank matrices $A$ and $B$ during training, ensuring that the adjustments are proportional to the original weight matrix's scale. This proportional adjustment improves the model's ability to fine-tune efficiently and effectively.

#### Usage in Training

During training, the low-rank matrices $A$ and $B$ are updated dynamically based on the magnitude vector $m$ and the directional component $V$. This dynamic adjustment allows the model to adapt more flexibly to new tasks, improving performance and reducing overfitting.

- **Weights Updated**: Similar to LoRA, only the low-rank matrices $A$ and $B$ are updated, but they are dynamically adjusted during training.
- **Improvement**: The key improvement of DoRA over LoRA lies in its ability to selectively focus on 
  directional adjustments while allowing separate training of the magnitude component. This separation can lead to 
  more effective fine-tuning, as it mimics the nuanced adjustments observed in full fine-tuning (FT), potentially improving learning efficiency and stability.

#### Detailed Explanation of $m$ and Directional Component

1. **Magnitude Vector $m$**: 
   - The parameter $m$ is initialized based on the norm of the pretrained weight matrix $W$.
   - It serves as a scaling factor that adjusts the magnitude of the weight updates.

2. **Directional Component**:
   - The directional component is calculated by normalizing the sum of the original weights $W$ and the scaled output from the low-rank adaptation (LoRA) $BA$.
   - This normalization ensures that the updates are directionally aligned with the original weight matrix.

The new weights for the linear layer are then calculated by scaling the directional component with the parameter $m$. This process ensures that the updates are not only directionally aligned but also appropriately scaled, leading to more effective fine-tuning.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```