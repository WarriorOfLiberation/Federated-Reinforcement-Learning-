# 🧬 Federated Reinforcement Learning for Personalized HIV Treatment

<div align="center">

![HIV Treatment](https://img.shields.io/badge/Medical%20AI-HIV%20Treatment-red?style=for-the-badge)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-blue?style=for-the-badge)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement-Learning-green?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange?style=for-the-badge)

*Exploring the application of Federated Reinforcement Learning (FRL) to determine optimal personalized treatment strategies for HIV patients using a simulated environment.*

</div>

---

## 🎯 **Project Overview**

This project leverages **cutting-edge AI techniques** to develop personalized HIV treatment strategies while preserving patient privacy. By combining Reinforcement Learning with Federated Learning, we train intelligent agents that can select optimal antiretroviral drug combinations over time, maximizing patient health outcomes while minimizing viral load and drug costs.

### 🔑 **Key Features**
- 🏥 **Privacy-Preserving**: Federated learning keeps sensitive patient data localized
- 🤖 **AI-Driven**: PPO reinforcement learning agent for dynamic treatment decisions
- 🧪 **Realistic Simulation**: Differential equation-based HIV patient environment
- 📊 **Comprehensive Evaluation**: Side-by-side comparison of federated vs standalone approaches

---

## 🌐 **The HIV Patient Environment**

Our simulation is built on the `HIVPatient-v0` Gymnasium environment, modeling the complex dynamics of HIV infection and treatment using established differential equations.

### 📈 **State Space** (6-dimensional vector)
| Component | Description | Role |
|-----------|-------------|------|
| `T1` | Healthy CD4+ T cells | Primary immune targets |
| `T1*` | Infected CD4+ T cells | Disease progression indicator |
| `T2` | Healthy macrophages | Secondary immune targets |
| `T2*` | Infected macrophages | Viral reservoir |
| `V` | Free virus particles | **Viral load** (key clinical metric) |
| `E` | Immune effector cells | Natural immune response |

### 💊 **Action Space** (Discrete medication decisions)
| Action | Treatment | Description |
|--------|-----------|-------------|
| `0` | 🚫 No drugs | No medication administered |
| `1` | 🔵 PI only | Protease Inhibitor |
| `2` | 🟢 RTI only | Reverse Transcriptase Inhibitor |
| `3` | 🔵🟢 Both PI + RTI | Combination therapy |

### 🏆 **Reward Function**
The agent is incentivized through a carefully designed reward system:
- ✅ **Positive rewards** for increasing healthy T1 and T2 cells
- ❌ **Penalties** for infected cells (T1*, T2*) and viral load (V)
- 💰 **Cost penalties** for medication usage (representing side effects/costs)

### 🎯 **Episode Termination**
Episodes end when:
- Maximum number of steps is reached (`max_episode_steps`)
- Patient achieves clinically stable state (low viral load, low infected cell count)

---

## 🤖 **Reinforcement Learning Agent (PPO)**

Our **Proximal Policy Optimization (PPO)** agent uses a sophisticated dual-head neural network architecture optimized for continuous state spaces and discrete action spaces.

### 🏗️ **Network Architecture**
```
Input (6D State) → Hidden Layers (ReLU) → Actor Head (Policy Distribution)
                                       ↘ Critic Head (Value Estimation)
```

#### 🎭 **Actor Network**
- Takes patient state as input
- Outputs probability distribution over 4 medication actions
- Defines the agent's treatment policy

#### 🎯 **Critic Network**
- Takes patient state as input
- Estimates expected cumulative future reward
- Improves learning stability through value-based guidance

### ⚡ **Learning Process**
1. **Data Collection**: Agent interacts with environment, storing experiences in memory buffer
2. **Return Calculation**: Computes discounted future rewards and advantage estimates
3. **Policy Update**: Uses PPO's clipped surrogate objective for stable parameter updates
4. **Value Update**: Critic trained using Mean Squared Error (MSE) loss
5. **Exploration**: Entropy bonus encourages diverse action exploration

---

## 🌍 **Federated Learning Framework (Flower)**

### 🖥️ **Server Architecture**
- **🎯 Central Coordinator**: Manages global model aggregation using `FedAvgWithMetrics` strategy
- **📊 Metrics Collection**: Tracks and stores training/evaluation metrics from all clients
- **💾 Model Persistence**: Saves aggregated global model weights periodically
- **🔄 Communication**: Orchestrates federated rounds between clients

### 👥 **Client Architecture**
Each client represents a healthcare institution with:

#### 🏥 **Local Data Management**
- **Patient Simulation**: Generates local sets of simulated HIV patients
- **Non-IID Distribution**: Patient parameters clustered around client-specific baselines
- **Realistic Diversity**: Reflects variations in patient populations across healthcare settings

#### 🔒 **Privacy Preservation**
- **Local Training**: PPO agents trained exclusively on institutional data
- **Parameter Sharing**: Only model weights (not raw data) shared with server
- **Decentralized Approach**: Raw patient information never leaves client environment

#### 🔄 **Federated Operations**
- **`fit()` Phase**: Receive global weights → Train locally → Send updated weights
- **`evaluate()` Phase**: Test global model on local patients → Report metrics
- **Weight Management**: Handles parameter extraction/loading for Flower integration

---

## 📊 **Training Progress**

### Training and Evaluation Rewards
*[Image placeholder - Training rewards and evaluation rewards over episodes]*

The training progress shows the learning curve of our PPO agent, with rewards improving over episodes as the agent learns optimal treatment policies.

---

## 🔬 **Methodology & Evaluation**

### 🎯 **Training Approaches**

#### 1️⃣ **Standalone Training** (Baseline)
- Single PPO agent trained centrally
- Access to large, diverse pool of simulated patients
- Random sampling from patient pool for training episodes
- Represents traditional centralized ML approach

#### 2️⃣ **Federated Training** (Novel Approach)
- Multiple clients with local PPO agent clones
- Each client trains on smaller, local patient datasets
- Periodic weight aggregation using FedAvg algorithm
- Simulates real-world distributed healthcare scenario

### 📈 **Evaluation Framework**
- **Fair Comparison**: Both models tested on identical set of unseen test patients
- **Performance Metrics**: Average cumulative reward over multiple episodes
- **Generalization Assessment**: Measures how well models handle previously unseen patient cases

---

## 🎨 **Results Visualization**

### Medication Actions Over Time
![image](https://github.com/user-attachments/assets/a152f6d7-f14a-4b7f-ab22-b47412a58681)


Both federated and standalone agents learned dynamic treatment strategies, primarily alternating between RTI-only therapy and combination PI+RTI treatment, demonstrating sophisticated policy learning.

### Reward Comparison
![image](https://github.com/user-attachments/assets/0fed56ff-adef-45ec-a03c-c3ed1a480d62)


**Performance Summary:**
- **Standalone Model**: Total reward = 48,712.36
- **Global Federated Model**: Total reward = 46,220.28
- **Outcome**: Both models achieved substantial rewards with comparable performance

### Viral Load Trajectory
![image](https://github.com/user-attachments/assets/474d37ac-1b1f-40ec-a400-d2b88476b40f)


Both models successfully reduced viral load from initial high levels (~10⁵) to significantly lower, controlled levels, demonstrating effective treatment policies.

---

## 🏆 **Key Findings**

### ✅ **Successful Federated Learning**
- Federated approach achieved **comparable performance** to centralized training
- Global model learned effective, dynamic treatment policies
- Privacy preservation maintained without significant performance cost

### 🎯 **Treatment Strategy Insights**
- Both models converged to **aggressive intervention strategies**
- Preference for combination therapy (PI + RTI) when needed
- Strategic use of RTI-only periods for maintenance therapy

### 📊 **Clinical Relevance**
- Significant **viral load reduction** achieved by both approaches
- High cumulative rewards indicate **clinically meaningful outcomes**
- Dynamic policies adapt to changing patient conditions over time

### 🔒 **Privacy Benefits**
- Patient data remains **localized to institutions**
- Effective learning achieved through **parameter sharing only**
- Viable path for **real-world deployment** in healthcare networks

---

## 🚀 **Impact & Future Directions**

This research demonstrates that **Federated Reinforcement Learning** is a viable approach for developing personalized HIV treatment strategies while maintaining patient privacy. The results provide evidence that:

- 🏥 Healthcare institutions can collaborate on AI model development without sharing sensitive patient data
- 🤖 RL agents can learn complex, dynamic treatment policies in federated settings
- 📈 Performance remains competitive with centralized approaches
- 🌍 Scalable framework for distributed medical AI applications

### 🔮 **Next Steps**
- Integration with real-world clinical data
- Extension to other chronic disease management scenarios
- Advanced federated learning algorithms (FedProx, FedNova)
- Multi-objective optimization for treatment personalization

---

<div align="center">

**🔬 Built with Science • 🤖 Powered by AI • 🔒 Privacy by Design**

*This project represents a step forward in privacy-preserving medical AI, demonstrating how federated learning can enable collaborative healthcare innovation while protecting patient confidentiality.*

</div>
