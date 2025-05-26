# Federated-Reinforcement-Learning

# Federated Reinforcement Learning for Personalized HIV Treatment

This project explores the application of Federated Reinforcement Learning (FRL) to determine optimal personalized treatment strategies for HIV patients using a simulated environment. The goal is to train a reinforcement learning agent that can select the best combination of antiretroviral drugs over time, aiming to maximize patient health (measured by a reward function) while minimizing viral load and drug costs. Federated Learning is employed to allow training on decentralized patient data, respecting data privacy and confidentiality.

## The HIV Patient Environment

The core of the project is a simulated environment representing the dynamics of HIV infection and treatment within a patient. This is implemented as a custom Gymnasium (formerly OpenAI Gym) environment, `HIVPatient-v0`, based on established differential equation models describing the interaction between HIV and the human immune system.

*   **State Space:** The environment state is a 6-dimensional vector representing the concentrations of key cell types and the virus:
    *   `T1`: Healthy CD4+ T cells (target cells)
    *   `T1*`: Infected CD4+ T cells
    *   `T2`: Healthy macrophages (another target cell type)
    *   `T2*`: Infected macrophages
    *   `V`: Free virus particles (viral load)
    *   `E`: Immune effector cells (immune system response)
    The state variables are non-negative floating-point numbers.
*   **Action Space:** The agent's action space is discrete, representing different medication actions:
    *   `0`: No drugs administered
    *   `1`: Administer Protease Inhibitor (PI) only
    *   `2`: Administer Reverse Transcriptase Inhibitor (RTI) only
    *   `3`: Administer both PI and RTI
*   **Dynamics:** The environment simulates the change in state variables over time (steps, representing days) based on a system of differential equations. These equations incorporate factors like cell production and death rates, infection rates by the virus, virus production by infected cells, and the immune response. Patient-specific biological parameters (e.g., `k1`, `k2`, `f`) influence these dynamics, leading to varied responses among individuals.
*   **Reward Function:** The agent receives a reward at each step designed to incentivize clinically desirable outcomes:
    *   Positive reward for increasing healthy T1 and T2 cells.
    *   Penalty for increasing infected T1* and T2* cells.
    *   Penalty for increasing viral load (V).
    *   Penalty for taking medication (actions 1, 2, or 3) to represent drug costs or side effects.
    The specific weights for these components can be tuned. The reward is scaled down (`reward_scale`) for numerical stability.
*   **Episode Termination:** An episode can end if a maximum number of steps is reached (`max_episode_steps`) or if the patient achieves a state considered clinically stable (low viral load, low infected cell count).

## Reinforcement Learning Agent (PPO)

A Proximal Policy Optimization (PPO) agent is implemented using PyTorch. PPO is a policy gradient method that is well-suited for environments with continuous state spaces and discrete action spaces.

*   **Policy Network:** The agent uses a single neural network with two heads: an *actor* and a *critic*.
    *   The **actor** takes the patient's state as input and outputs a probability distribution over the four possible medication actions. This distribution defines the agent's *policy*.
    *   The **critic** takes the patient's state as input and outputs an estimate of the *value* of being in that state (i.e., the expected cumulative future reward from that state).
    The network consists of two fully connected hidden layers with ReLU activation.
*   **Learning Process:**
    *   The agent interacts with the environment for a certain number of steps (or an episode), collecting data (state, action, reward, etc.) in a `Memory` buffer.
    *   After collecting a trajectory, the agent computes the discounted future rewards (returns) and an estimate of the advantage for each action taken (how much better or worse the action was than expected by the critic).
    *   The policy network's parameters are updated using the PPO algorithm, which optimizes a "clipped" surrogate objective function to take large training steps without straying too far from the old policy, ensuring stability.
    *   Simultaneously, the critic head is updated using Mean Squared Error (MSE) to improve its value estimation.
    *   An entropy bonus is added to the loss to encourage exploration.
*   **Weight Handling:** The agent includes `get_weights` and `set_weights` methods specifically designed to extract and load the neural network parameters as NumPy arrays, including handling parameter shapes (transposing weights as needed), which is required for integration with the Flower Federated Learning framework.

## Federated Learning Framework (Flower)

The project utilizes the Flower framework to implement federated learning.

*   **Server:** A central server coordinates the training process. It uses a custom strategy, `FedAvgWithMetrics`, inheriting from Flower's standard `FedAvg`. This strategy performs Federated Averaging, aggregating the model weights received from multiple clients. Crucially, it also collects and stores training and evaluation metrics reported by the clients and saves the aggregated global model weights periodically.
*   **Clients:** Each client is an instance of the `HIVClient` class.
    *   Each client represents a potential data silo, such as a hospital or clinic, and holds a local set of simulated HIV patients.
    *   The clients implement Flower's `NumPyClient` interface, defining methods to `get_parameters` (send local model weights), `fit` (receive global weights, train locally on its patients, send updated weights), and `evaluate` (receive global weights, evaluate on local patients, send evaluation metrics).
    *   Importantly, the `generate_local_patients` method for clients can be configured to simulate non-IID (non-identically and independently distributed) data by generating patient parameters that are clustered around a client-specific baseline, reflecting variations in patient populations across different healthcare settings.
    *   During the `fit` phase, a client receives the current global model, updates its local PPO agent's weights, trains this agent on its *own* patients for a specified number of local epochs, and then sends the updated local weights back to the server for aggregation.
    *   During the `evaluate` phase, a client evaluates the current global model (downloaded from the server) on its local patients and reports the performance metrics.

## Methodology

The project implements and compares two training approaches:

1.  **Standalone Training:** A single PPO agent is trained in a centralized manner. This agent has access to a relatively large pool of diverse simulated patients, sampling randomly from this pool for training episodes. This represents the traditional centralized training approach and serves as a baseline for comparison.
2.  **Federated Training:** The federated learning simulation is run using Flower. A central server coordinates multiple clients. Each client trains a local PPO agent clone on its smaller, local patient dataset. Periodically, clients send their model updates to the server, which aggregates them (using FedAvg) to create a new global model version, which is then sent back to the clients for the next round. This process is repeated for a set number of communication rounds.

**Evaluation and Comparison:**

*   Both the final standalone model and the global federated model are evaluated on a separate, consistent set of **test patients** that were *not* used during the training phases. This ensures a fair comparison of how well each trained model generalizes to unseen patients. Performance is measured by the average cumulative reward achieved over multiple episodes on these test patients.
*   To gain deeper insight into *how* the models make decisions and their immediate impact, a `compare_trajectories` function is used. This function takes a patient environment and a dictionary of trained models (e.g., the Global FL model and the Standalone model) and runs a single episode for each model on that *specific* patient. It records the full sequence of states, actions, and rewards, generating plots that visualize:
    *   Medication actions taken over time.
    *   Cumulative reward progression over time.
    *   Viral load changes over time.
    *   Other relevant state variables (like T-cell counts) over time.

## Results

The plots generated by the comparison function illustrate the learned policies and their outcomes for a single representative patient when treated by the Global Federated model and the Standalone model.

### Medication Actions Over Time (Plot 1)

*   The plot shows the sequence of medication actions taken by both the "Global" and "Standalone" agents over 100 steps.
*   Both agents learned a highly dynamic treatment strategy, primarily oscillating between Action 2 (RTI only) and Action 3 (Both PI and RTI). Action 0 (No Drugs) and Action 1 (PI only) are used less frequently or not at all in this trajectory.
*   This suggests that both the federated and standalone training processes converged to a similar high-level understanding of the treatment strategy, favoring aggressive intervention combinations when needed, followed by periods of potentially less intense therapy (RTI only).

### Reward Comparison (Plot 2)

*   This plot shows the *cumulative reward* obtained by each model over the 100 steps for the specific patient trajectory. A higher cumulative reward indicates better performance according to the defined reward function.
*   Both models achieve substantial cumulative rewards, indicating successful treatment based on the environment's criteria.
*   On this particular test patient, the "Standalone" model (Total=48712.36) achieved a slightly higher final cumulative reward than the "Global" model (Total=46220.28). The curve shapes are similar, showing relatively rapid reward increase initially, followed by a slower, steadier gain.

### Viral Load Comparison (Plot 3)

*   This plot visualizes the *viral load* over time, a critical clinical metric. The y-axis is on a logarithmic scale, which is standard for viral load measurements.
*   Starting from a high viral load (unhealthy state, around 10<sup>5</sup>), both models successfully reduce the viral load significantly over the 100 steps, bringing it down towards lower levels.
*   Consistent with the reward comparison, the "Standalone" model appears to achieve a slightly more favorable viral load trajectory on this specific patient, potentially reaching lower levels or suppressing the virus more effectively compared to the "Global" model during this episode. The initial "spike" observed might be related to the environmental dynamics during the initial phase of recovery or drug uptake delay, which is then followed by a significant drop due to treatment effect.

**Summary of Results from Provided Plots:**

The plots for this specific comparison patient demonstrate that the Federated Learning approach successfully trained an agent that learned a dynamic and effective treatment policy, comparable to a standalone agent trained on centralized data. While the standalone agent showed a slight edge in terms of cumulative reward and viral load reduction *on this single patient*, the overall outcomes achieved by the Federated model were very close and clinically meaningful (significant viral load reduction, high reward). This provides evidence that FRL is a viable method for learning effective, personalized HIV treatment policies while potentially preserving patient data privacy by keeping sensitive patient information localized to clients.

*(Note: The full evaluation across multiple test patients, saved in `results/server/global_evaluation.json`, would provide a more robust comparison of the average performance between the Global and Standalone models.)*
