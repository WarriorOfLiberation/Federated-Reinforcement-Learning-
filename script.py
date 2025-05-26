import gymnasium as gym
import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
import traceback
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import copy
import random
from collections import deque
import os
import time
import json
from tqdm import tqdm
import flwr as fl
import argparse
from typing import Dict, List, Tuple, Optional, Union
import pickle
import threading
import multiprocessing as mp

# Register the environment
from gymnasium.envs.registration import register

register(
    id="HIVPatient-v0",
    entry_point="__main__:HIVPatient",
)

# HIV Patient Environment
class HIVPatient(gym.Env):
    """
    HIV Patient Environment based on the paper:
    "Optimizing Treatment of HIV Patients"
    """
    def __init__(self):
        super(HIVPatient, self).__init__()
        
        # Define action and observation space
        # Actions: 0 = no drugs, 1 = PI only, 2 = RTI only, 3 = both drugs
        self.action_space = gym.spaces.Discrete(4)
        
        # State space: 6 parameters (T1, T1*, T2, T2*, V, E)
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Patient parameters (can be customized for different patients)
        self.lambda1 = 10000  # production rate for healthy T1 cells
        self.lambda2 = 31.98  # production rate for healthy T2 cells
        self.d1 = 0.01  # death rate for healthy T1 cells
        self.d2 = 0.01  # death rate for healthy T2 cells
        self.f = 0.34  # fraction of virions infecting T2 (vs T1)
        self.k1 = 8e-7  # infection rate for T1
        self.k2 = 1e-4  # infection rate for T2
        self.delta = 0.7  # death rate for infected cells
        self.m1 = 1e-5  # immune-induced clearance rate for infected T1
        self.m2 = 1e-5  # immune-induced clearance rate for infected T2
        self.NT = 100  # virions produced per infected cell
        self.c = 13  # virus natural death rate
        self.rho1 = 1  # likelihood of T1 cell killing an infected cell
        self.rho2 = 1  # likelihood of T2 cell killing an infected cell
        self.lambdaE = 1  # production rate of immune effector cells
        self.bE = 0.3  # maximum birth rate for immune cells
        self.Kb = 100  # saturation constant for immune cell birth
        self.dE = 0.25  # maximum death rate for immune cells
        self.Kd = 500  # saturation constant for immune cell death
        self.deltaE = 0.1  # natural death rate for immune cells
        
        # Drug efficacy parameters
        self.epsilonRT = 0.7  # RTI efficacy
        self.epsilonPI = 0.3  # PI efficacy
        
        # State variables
        self.T1 = 0  # healthy type 1 cells
        self.T1star = 0  # infected type 1 cells
        self.T2 = 0  # healthy type 2 cells
        self.T2star = 0  # infected type 2 cells
        self.V = 0  # free virus
        self.E = 0  # immune effector cells
        
        # Reward parameters
        self.reward_scale = 1e-2  # scaling factor for rewards
        
        # Episode counter
        self.episode_step = 0
        self.max_episode_steps = 200
        
    def reset(self, seed=None, options=None, mode="random"):
        """
        Reset the environment to an initial state.
        mode: 'random', 'healthy', or 'unhealthy'
        """
        super().reset(seed=seed)
        
        # Reset episode step counter
        self.episode_step = 0
        
        if mode == "healthy":
            # Healthy patient baseline
            self.T1 = 1000000
            self.T1star = 0
            self.T2 = 3198
            self.T2star = 0
            self.V = 0
            self.E = 10
        elif mode == "unhealthy":
            # Unhealthy patient baseline (high viral load)
            self.T1 = 163573
            self.T1star = 11945
            self.T2 = 5
            self.T2star = 46
            self.V = 63919
            self.E = 24
        else:  # random
            # Random initialization
            self.T1 = np.random.uniform(100000, 1000000)
            self.T1star = np.random.uniform(0, 20000)
            self.T2 = np.random.uniform(5, 5000)
            self.T2star = np.random.uniform(0, 100)
            self.V = np.random.uniform(0, 100000)
            self.E = np.random.uniform(10, 50)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get the current state observation"""
        return np.array([
            self.T1, self.T1star, self.T2, self.T2star, self.V, self.E
        ], dtype=np.float32)
    
    def step(self, action):
        """
        Take a step in the environment given an action.
        action: 0 = no drugs, 1 = PI only, 2 = RTI only, 3 = both drugs
        """
        # Increment episode step
        self.episode_step += 1
        
        # Apply the drug efficacy based on the action
        epsilon1 = 0  # RTI efficacy
        epsilon2 = 0  # PI efficacy
        
        if action == 1:  # PI only
            epsilon2 = self.epsilonPI
        elif action == 2:  # RTI only
            epsilon1 = self.epsilonRT
        elif action == 3:  # Both drugs
            epsilon1 = self.epsilonRT
            epsilon2 = self.epsilonPI
        
        # Calculate the derivatives
        dT1 = self.lambda1 - self.d1 * self.T1 - (1 - epsilon1) * self.k1 * self.V * self.T1
        dT1star = (1 - epsilon1) * self.k1 * self.V * self.T1 - self.delta * self.T1star - self.m1 * self.E * self.T1star
        dT2 = self.lambda2 - self.d2 * self.T2 - (1 - self.f) * (1 - epsilon1) * self.k2 * self.V * self.T2
        dT2star = (1 - self.f) * (1 - epsilon1) * self.k2 * self.V * self.T2 - self.delta * self.T2star - self.m2 * self.E * self.T2star
        dV = (1 - epsilon2) * self.NT * self.delta * (self.T1star + self.T2star) - self.c * self.V - ((1 - epsilon1) * self.rho1 * self.k1 * self.T1 + (1 - epsilon1) * (1 - self.f) * self.rho2 * self.k2 * self.T2) * self.V
        dE = self.lambdaE + self.bE * (self.T1star + self.T2star) * self.E / (self.T1star + self.T2star + self.Kb) - self.dE * (self.T1star + self.T2star) * self.E / (self.T1star + self.T2star + self.Kd) - self.deltaE * self.E
        
        # Update the state variables using Euler's method with a time step of 1 day
        self.T1 = max(0, self.T1 + dT1)
        self.T1star = max(0, self.T1star + dT1star)
        self.T2 = max(0, self.T2 + dT2)
        self.T2star = max(0, self.T2star + dT2star)
        self.V = max(0, self.V + dV)
        self.E = max(0, self.E + dE)
        
        # Calculate reward
        # The reward is based on the health status of the patient
        # We want to maximize healthy cells and minimize infected cells and virus
        reward = self.reward_scale * (
            self.T1 + self.T2  # reward for healthy cells
            - 0.1 * (self.T1star + self.T2star)  # penalty for infected cells
            - 0.01 * self.V  # penalty for virus
        )
        
        # Apply drug cost if used
        if action > 0:
            reward -= 0.5 * self.reward_scale  # drug cost
        
        # Check if the episode is done
        done = (self.episode_step >= self.max_episode_steps) or (self.V <= 50 and self.T1star <= 50 and self.T2star <= 50)
        
        # Truncate if we've reached the maximum number of steps
        truncated = self.episode_step >= self.max_episode_steps
        
        return self._get_observation(), reward, done, truncated, {}

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value

# Define PPO Agent
class PPO:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, clip_ratio=0.2, epochs=10, agent_id=None):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.agent_id = agent_id
        
    def get_action(self, state, evaluation=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.policy(state)
        
        if evaluation:
            action = torch.argmax(action_probs, dim=1).item()
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            
        return action
    
    def evaluate(self, state, action):
        state = torch.FloatTensor(state)
        action_probs, value = self.policy(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(torch.tensor(action))
        entropy = dist.entropy()
        
        return action_logprobs, value, entropy
    
    def update(self, memory):
        # Convert list to tensor
        old_states = torch.FloatTensor(memory.states)
        old_actions = torch.LongTensor(memory.actions)
        old_logprobs = torch.FloatTensor(memory.logprobs)
        old_rewards = torch.FloatTensor(memory.rewards)
        old_values = torch.FloatTensor(memory.values)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        for r, v in zip(reversed(old_rewards), reversed(old_values)):
            R = r + self.gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - v.item())
        
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimize policy for K epochs
        for _ in range(self.epochs):
            # Evaluate actions
            logprobs, state_values, entropy = self.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            
            # Compute ratios
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
            
            # Final loss with value loss and entropy bonus
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_values, returns)
            entropy_loss = -entropy.mean() * 0.01
            
            loss = policy_loss + 0.5 * value_loss + entropy_loss
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def get_weights(self):
        """Get model weights as a list of NumPy arrays (with correct shape)"""
        ordered_keys = sorted(self.policy.state_dict().keys())
        weights = []
        for k in ordered_keys:
            param = self.policy.state_dict()[k].cpu().numpy()
            # Transpose only weight matrices (not bias vectors)
            if "weight" in k and param.ndim == 2:
                param = param.T
            weights.append(param)
        return weights

    def set_weights(self, weights):
        """Set model weights from a list of NumPy arrays (transposing if necessary)"""
        ordered_keys = sorted(self.policy.state_dict().keys())
        
        if len(ordered_keys) != len(weights):
            raise ValueError(f"Parameter count mismatch: model has {len(ordered_keys)} parameters, but received {len(weights)}")

        state_dict = OrderedDict()
        for i, (k, w) in enumerate(zip(ordered_keys, weights)):
            model_shape = self.policy.state_dict()[k].shape
            weight_shape = w.shape

            # Transpose back if it's a weight matrix
            if "weight" in k and len(model_shape) == 2:
                w = w.T
                weight_shape = w.shape  # Update for shape check
            
            if model_shape != weight_shape:
                raise ValueError(f"Shape mismatch for {k}: model expects {model_shape}, but received {weight_shape}")
            
            state_dict[k] = torch.tensor(w, dtype=self.policy.state_dict()[k].dtype)

        self.policy.load_state_dict(state_dict, strict=True)

    
    def print_model_info(self):
        """Print model information for debugging"""
        print(f"Model architecture: {self.policy}")
        print(f"Model parameters:")
        for name, param in self.policy.named_parameters():
            print(f"  {name}: {param.shape}")
        
    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)
        
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))

# Memory buffer for storing trajectories
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.is_terminals[:]
        
    def add(self, state, action, logprob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.is_terminals.append(done)

# Function to train an agent on a specific patient
def train_on_patient(agent, env, patient_params, max_steps=200):
    """Train an agent on a single patient for one episode"""
    memory = Memory()
    episode_reward = 0
    
    # Reset the environment
    state, info = env.reset(options={"mode": "unhealthy"})        # Set patient-specific parameters

    
    # Apply patient-specific parameters
    if patient_params:
        env.k1 = patient_params["k1"]
        env.k2 = patient_params["k2"]
        env.f = patient_params["f"]
    
    for step in range(max_steps):
        # Get action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, value = agent.policy(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample().item()
            logprob = dist.log_prob(torch.tensor(action)).item()
        
        # Take action in environment
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Save in memory
        memory.add(state, action, logprob, reward, value, done or truncated)
        
        state = next_state
        episode_reward += reward
        
        if done or truncated:
            break
    
    # Update the agent using collected experience
    loss = agent.update(memory)
    
    return episode_reward, loss

# Function to evaluate an agent on a patient
def evaluate_patient(agent, env, patient_params, num_episodes=5, max_steps=200):
    """Evaluate an agent on a single patient"""
    rewards = []
    
    for _ in range(num_episodes):
        total_reward = 0
        state, info = env.reset(options={"mode": "unhealthy"})        # Set patient-specific parameters

        
        # Set patient-specific parameters
        if patient_params:
            env.k1 = patient_params["k1"]
            env.k2 = patient_params["k2"]
            env.f = patient_params["f"]
            
        for _ in range(max_steps):
            # Select action with evaluation mode
            action = agent.get_action(state, evaluation=True)
            
            # Execute action
            next_state, reward, done, truncated, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards)

# Function to generate patient parameters
def generate_patient(patient_id):
    """Generate a patient with random parameters"""
    k1 = np.random.uniform(5e-7, 8e-7)
    k2 = np.random.uniform(0.1e-4, 1.0e-4)
    f = np.random.uniform(0.29, 0.34)
    
    return {
        "id": patient_id,
        "params": {
            "k1": k1,
            "k2": k2,
            "f": f
        }
    }

# Function to generate test patients
def generate_test_patients(num_patients=10, seed=42):
    """Generate a set of test patients with consistent parameters"""
    np.random.seed(seed)
    test_patients = []
    for i in range(num_patients):
        test_patients.append(generate_patient(f"test_patient_{i}"))
    np.random.seed(None)  # Reset seed
    return test_patients

# Flower Client Implementation
class HIVClient(fl.client.NumPyClient):
    def __init__(self, client_id, num_patients=5, seed=None):
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        # Save client ID
        self.client_id = client_id
        self.name = f"client_{client_id}"
        
        # Create environment
        self.env = gym.make("HIVPatient-v0")
        
        # Generate patients for this client
        self.patients = self.generate_local_patients(num_patients)
        
        # Create agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.agent = PPO(state_dim, action_dim, agent_id=self.name)
        
        # Create results directory
        self.results_dir = f"results/client_{client_id}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Save patient information
        with open(f"{self.results_dir}/patients.json", "w") as f:
            json.dump([{
                "id": p["id"],
                "params": p["params"],
            } for p in self.patients], f, indent=2)
        
        print(f"Client {client_id} initialized with {num_patients} patients")
    
    def generate_local_patients(self, num_patients):
        """Generate patients for this client with specific characteristics"""
        # Create base parameters that could define the client's "specialty"
        # For example, clients might specialize in certain patient populations
        base_k1 = np.random.uniform(0.85, 1.15)  # Base immune system strength for this client
        base_k2 = np.random.uniform(0.85, 1.15)  # Base virus replication rate for this client
        base_f = np.random.uniform(0.85, 1.15)   # Base drug efficacy for this client
        
        patients = []
        
        for i in range(num_patients):
            # Generate patient with slight variations around the base parameters
            k1 = base_k1 * np.random.uniform(0.95, 1.05)
            k2 = base_k2 * np.random.uniform(0.95, 1.05)
            f = base_f * np.random.uniform(0.95, 1.05)
            
            # Ensure parameters are within valid ranges
            k1 = max(0.5, min(k1, 1.5))
            k2 = max(0.5, min(k2, 1.5))
            f = max(0.5, min(f, 1.5))
            
            patient = {
                "id": f"{self.name}_patient_{i}",
                "params": {
                    "k1": k1,
                    "k2": k2,
                    "f": f
                }
            }
            
            patients.append(patient)
        
        return patients
    
    def get_parameters(self, config):
        """Get client model parameters as a list of NumPy arrays"""
        try:
            weights = self.agent.get_weights()
            # Print debug info about the weights being sent
            print(f"Client {self.client_id} sending {len(weights)} parameters")
            return weights
        except Exception as e:
            print(f"Error in get_parameters: {e}")
            raise
    
    def fit(self, parameters, config):
        """Train client model on local data"""
        print(f"Client {self.client_id} training with {len(parameters)} parameters")
        
        try:
            # Set weights received from server
            self.agent.set_weights(parameters)
            
            # Train for specified number of local epochs
            local_epochs = config.get("local_epochs", 5)
            
            # Track training metrics
            mean_loss = 0.0
            mean_reward = 0.0
            
            for _ in range(local_epochs):
                for patient in self.patients:
                    # Train on this patient for one episode
                    reward, loss = train_on_patient(
                        self.agent, self.env, patient["params"], max_steps=200
                    )
                    
                    mean_loss += loss
                    mean_reward += reward
            
            # Compute means
            mean_loss /= (local_epochs * len(self.patients))
            mean_reward /= (local_epochs * len(self.patients))
            
            # Save local model
            self.save_model()
            
            # Return updated model parameters and metrics
            return self.get_parameters(config), len(self.patients), {
                "train_loss": float(mean_loss),
                "train_reward": float(mean_reward)
            }
        except Exception as e:
            print(f"Error in fit method of client {self.client_id}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def evaluate(self, parameters, config):
        """Evaluate client model"""
        print(f"Client {self.client_id} evaluating with {len(parameters)} parameters")
        
        try:
            # Set weights received from server
            self.agent.set_weights(parameters)
            
            # Evaluate on local patients
            val_loss = 0.0
            val_reward = 0.0
            
            for patient in self.patients:
                # Evaluate on this patient
                reward, _ = evaluate_patient(
                    self.agent, self.env, patient["params"], num_episodes=3
                )
                
                val_reward += reward
            
            # Compute means
            val_reward /= len(self.patients)
            
            return 0.0, len(self.patients), {
                "val_loss": 0.0,
                "val_reward": float(val_reward)
            }
        except Exception as e:
            print(f"Error in evaluate method of client {self.client_id}: {e}")
            import traceback
            traceback.print_exc()
            raise


    def save_model(self):
        """Save client model"""
        self.agent.save_model(f"{self.results_dir}/model.pt")



def generate_initial_parameters(state_dim=6, action_dim=4):
    """Generate initial parameters for the model to ensure consistent initialization"""
    # Create a temporary model
    model = PPO(state_dim, action_dim)
    
    # Get the parameters
    parameters = model.get_weights()
    
    # Print debug information
    print("Generated initial parameters:")
    print(f"Number of parameters: {len(parameters)}")
    for i, param in enumerate(parameters):
        print(f"  Parameter {i} shape: {param.shape}")
        
    return parameters


# Custom Flower Strategy for Federated Averaging with Metrics Tracking
class FedAvgWithMetrics(fl.server.strategy.FedAvg):
    """Federated Averaging with metrics collection"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = None
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "train_reward": [],
            "val_reward": []
        }

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model weights and training metrics"""
        # Call parent's aggregate_fit for weights aggregation
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Save the aggregated parameters
            self.parameters = aggregated_parameters[0]
            
            # Extract and save training metrics
            metrics = [(r.metrics["train_loss"], r.metrics["train_reward"]) for _, r in results]
            losses = [m[0] for m in metrics]
            rewards = [m[1] for m in metrics]
            
            if losses:
                self.metrics_history["train_loss"].append(sum(losses) / len(losses))
            if rewards:
                self.metrics_history["train_reward"].append(sum(rewards) / len(rewards))
                
            # Save global model after each round
            try:
                global_model = save_global_model(self.parameters)
                print(f"Round {server_round}: Global model saved successfully")
                test_patients = generate_test_patients(num_patients=3)  # Use fewer patients for quicker evaluation
                evaluate_global_model(global_model, test_patients)
            except Exception as e:
                print(f"Round {server_round}: Error saving global model: {e}")
            
        return aggregated_parameters

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics"""
        # Call parent's aggregate_evaluate
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # Extract and save validation metrics
        if results:
            metrics = [(r.metrics["val_loss"], r.metrics["val_reward"]) for _, r in results]
            losses = [m[0] for m in metrics]
            rewards = [m[1] for m in metrics]
            
            if losses:
                self.metrics_history["val_loss"].append(sum(losses) / len(losses))
            if rewards:
                self.metrics_history["val_reward"].append(sum(rewards) / len(rewards))
                
        return aggregated_metrics
    

from flwr.common import parameters_to_ndarrays  # Add this import at the top

def save_global_model(global_model_weights, state_dim=6, action_dim=4):
    """Save global model weights"""
    try:
        # Convert Flower Parameters object to list of NumPy arrays
        weights = parameters_to_ndarrays(global_model_weights)

        # Create global model
        global_model = PPO(state_dim, action_dim, agent_id="global")
        global_model.print_model_info()

        print(f"Global model has {len(global_model.get_weights())} parameters")
        print(f"Received {len(weights)} parameters from server")

        # Set weights
        global_model.set_weights(weights)

        # Save model
        os.makedirs("results/server", exist_ok=True)
        global_model.save_model("results/server/global_model.pt")

        return global_model
    except Exception as e:
        print(f"Error in save_global_model: {e}")
        import traceback
        traceback.print_exc()
        raise

# Function to evaluate global model on test patients
def evaluate_global_model(global_model, test_patients, num_episodes=50):
    """Evaluate global model on test patients"""
    env = gym.make("HIVPatient-v0")
    results = {}
    
    print("\nEvaluating global model on test patients:")
    for patient in test_patients:
        mean_reward, std_reward = evaluate_patient(
            global_model, env, patient["params"], num_episodes=num_episodes
        )
        results[patient["id"]] = {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward)
        }
        print(f"  Patient {patient['id']}: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Calculate overall metrics
    all_rewards = [results[patient_id]["mean_reward"] for patient_id in results]
    overall_mean = np.mean(all_rewards)
    overall_std = np.std(all_rewards)
    
    results["overall"] = {
        "mean_reward": float(overall_mean),
        "std_reward": float(overall_std)
    }
    
    print(f"Overall: {overall_mean:.2f} ± {overall_std:.2f}")
    
    # Save results
    os.makedirs("results/server", exist_ok=True)
    with open("results/server/global_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

# Function to compare trajectories between global and client models
import json
import numpy as np
import os
import matplotlib.pyplot as plt

def make_json_serializable(obj):
    """Convert NumPy data types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()  # Convert numpy scalar types to native Python types
    else:
        return obj

def compare_trajectories(models, env, patient_params, max_steps=100):
    """Compare trajectories of different models on the same patient"""
    trajectories = {}

    for model_name, model in models.items():
        # Reset environment
        state, info = env.reset(options={"mode": "unhealthy"})

        if patient_params:
            env.k1 = patient_params["k1"]
            env.k2 = patient_params["k2"]
            env.f = patient_params["f"]

        traj = {
            "states": [],
            "actions": [],
            "rewards": [],
            "viral_load": []
        }

        traj["states"].append(state.tolist())
        traj["viral_load"].append(state[4])

        total_reward = 0
        for step in range(max_steps):
            action = model.get_action(state, evaluation=True)
            next_state, reward, done, truncated, _ = env.step(action)

            traj["states"].append(next_state.tolist())
            traj["actions"].append(action)
            traj["rewards"].append(reward)
            traj["viral_load"].append(next_state[4])

            total_reward += reward
            state = next_state

            if done or truncated:
                break

        traj["total_reward"] = total_reward
        traj["steps"] = step + 1
        trajectories[model_name] = traj

    # Create output directory
    os.makedirs("results/comparisons", exist_ok=True)

    # === 1. Viral Load Over Time ===
    plt.figure(figsize=(12, 6))
    for model_name, traj in trajectories.items():
        plt.plot(traj["viral_load"], label=f"{model_name} (R={traj['total_reward']:.2f})")
    plt.xlabel("Steps")
    plt.ylabel("Viral Load")
    plt.title(f"Viral Load Comparison - Patient {patient_params.get('id', 'Unknown')}")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f"results/comparisons/viral_load_comparison_{patient_params.get('id', 'unknown')}.png")
    plt.close()

    # === 2. Cumulative Rewards ===
    plt.figure(figsize=(12, 6))
    for model_name, traj in trajectories.items():
        cumulative_rewards = np.cumsum(traj["rewards"])
        plt.plot(cumulative_rewards, label=f"{model_name} (Total={traj['total_reward']:.2f})")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Reward Comparison - Patient {patient_params.get('id', 'Unknown')}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/comparisons/reward_comparison_{patient_params.get('id', 'unknown')}.png")
    plt.close()

    # === 3. Medication Actions Over Time ===
    plt.figure(figsize=(12, 6))
    for model_name, traj in trajectories.items():
        plt.plot(traj["actions"], label=model_name)
    plt.xlabel("Steps")
    plt.ylabel("Medication Action")
    plt.title(f"Medication Actions Over Time - Patient {patient_params.get('id', 'Unknown')}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/comparisons/actions_over_time_{patient_params.get('id', 'unknown')}.png")
    plt.close()

    # === 4. Action Distribution Bar Plot ===
    action_labels = ["No Drug", "RTI", "PI", "Both"]
    model_names = list(trajectories.keys())
    action_counts = np.zeros((len(model_names), len(action_labels)))

    for i, model_name in enumerate(model_names):
        for a in trajectories[model_name]["actions"]:
            action_counts[i, a] += 1

    action_percents = action_counts / action_counts.sum(axis=1, keepdims=True) * 100
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(model_names))
    for i, label in enumerate(action_labels):
        plt.bar(model_names, action_percents[:, i], bottom=bottom, label=label)
        bottom += action_percents[:, i]
    plt.ylabel("Percentage of Time Each Action Was Taken")
    plt.title(f"Medication Action Distribution - Patient {patient_params.get('id', 'Unknown')}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/comparisons/action_distribution_{patient_params.get('id', 'unknown')}.png")
    plt.close()

    # === 5. Viral Load with Action Overlays ===
    plt.figure(figsize=(12, 6))
    for model_name, traj in trajectories.items():
        plt.plot(traj["viral_load"], label=model_name)
        actions = traj["actions"]
        for t, a in enumerate(actions):
            color = {1: "blue", 2: "green", 3: "red"}.get(a, None)
            if color:
                plt.axvline(x=t, color=color, alpha=0.1)
    plt.xlabel("Steps")
    plt.ylabel("Viral Load")
    plt.title(f"Viral Load with Actions Overlaid - Patient {patient_params.get('id', 'Unknown')}")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/comparisons/viral_load_with_actions_{patient_params.get('id', 'unknown')}.png")
    plt.close()

        # === 6. T-cell Count Over Time ===
    plt.figure(figsize=(12, 6))
    for model_name, traj in trajectories.items():
        t_cell_counts = [s[3] for s in traj["states"]]  # Assuming state[3] is T-cell count
        plt.plot(t_cell_counts, label=model_name)
    plt.xlabel("Steps")
    plt.ylabel("T-cell Count")
    plt.title(f"T-cell Count Over Time - Patient {patient_params.get('id', 'Unknown')}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/comparisons/tcell_over_time_{patient_params.get('id', 'unknown')}.png")
    plt.close()


        # === 7. Dual-Axis Plot: Viral Load and T-cell Count ===
    plt.figure(figsize=(12, 6))

    for model_name, traj in trajectories.items():
        steps = list(range(len(traj["viral_load"])))
        t_cells = [s[3] for s in traj["states"]]  # Assuming state[3] is T-cell count
        viral_load = traj["viral_load"]

        ax1 = plt.gca()
        ax1.plot(steps, viral_load, label=f"{model_name} - Viral Load", linestyle='--')
        ax1.set_ylabel("Viral Load (log scale)", color="blue")
        ax1.set_yscale("log")
        ax1.tick_params(axis='y', labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.plot(steps, t_cells, label=f"{model_name} - T-cells", linestyle='-')
        ax2.set_ylabel("T-cell Count", color="green")
        ax2.tick_params(axis='y', labelcolor="green")

    plt.title(f"Viral Load & T-cell Count - Patient {patient_params.get('id', 'Unknown')}")
    ax1.set_xlabel("Steps")
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/comparisons/dual_viral_tcell_{patient_params.get('id', 'unknown')}.png")
    plt.close()



    # Save trajectory data as JSON
    serializable_trajectories = make_json_serializable(trajectories)
    with open(f"results/comparisons/trajectories_{patient_params.get('id', 'unknown')}.json", "w") as f:
        json.dump(serializable_trajectories, f, indent=2)

    return trajectories

# Function to run a standalone PPO agent (no federation)
def train_standalone_agent(env, num_episodes=1000, max_steps=200, eval_interval=100):
    """Train a standalone PPO agent without federation"""
    print("Training standalone agent...")
    
    # Initialize environment and agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, action_dim, agent_id="standalone")
    
    # Generate a few patients for training
    patients = [generate_patient(f"standalone_patient_{i}") for i in range(10)]
    test_patients = generate_test_patients(num_patients=5)
    
    # Training history
    history = {
        "episode_rewards": [],
        "losses": [],
        "eval_rewards": [],
        "eval_episodes": []
    }
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training standalone agent")
    
    # Train for specified number of episodes
    for episode in pbar:
        # Select a random patient
        patient = random.choice(patients)
        
        # Train on patient
        episode_reward, loss = train_on_patient(
            agent, env, patient["params"], max_steps=max_steps
        )
        
        # Record history
        history["episode_rewards"].append(episode_reward)
        history["losses"].append(loss)
        
        # Update progress bar
        pbar.set_postfix({"reward": f"{episode_reward:.2f}", "loss": f"{loss:.4f}"})
        
        # Evaluate periodically
        if (episode + 1) % eval_interval == 0:
            eval_rewards = []
            for test_patient in test_patients:
                reward, _ = evaluate_patient(
                    agent, env, test_patient["params"], num_episodes=3
                )
                eval_rewards.append(reward)
            
            mean_eval_reward = np.mean(eval_rewards)
            history["eval_rewards"].append(mean_eval_reward)
            history["eval_episodes"].append(episode)
            
            print(f"\nEpisode {episode+1}: Eval reward: {mean_eval_reward:.2f}")
    
    # Save training history
    os.makedirs("results/standalone", exist_ok=True)
    with open("results/standalone/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save model
    agent.save_model("results/standalone/model.pt")
    
    # Plot training curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["episode_rewards"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["eval_episodes"], history["eval_rewards"])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Evaluation Rewards")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/standalone/training_curves.png")
    plt.close()
    
    return agent

# Function to start the Flower server
def start_server(num_rounds=10, num_clients=5, client_fraction=1.0):
    """Start the Flower server with the specified parameters"""
    print(f"Starting server with {num_rounds} rounds, expecting {num_clients} clients")
    
    # Define evaluation strategy with our improved class
    strategy = FedAvgWithMetrics(
        fraction_fit=client_fraction,
        fraction_evaluate=client_fraction,
        min_fit_clients=max(1, int(num_clients * client_fraction)),
        min_evaluate_clients=max(1, int(num_clients * client_fraction)),
        min_available_clients=num_clients,
        # Add initial parameters if needed
        initial_parameters=None
    )
    
    # Generate test patients for global evaluation
    test_patients = generate_test_patients(num_patients=10)
    
    # Start Flower server
    try:
        history = fl.server.start_server(
            server_address="0.0.0.0:8089",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy
        )
        
        # Try to get final weights
        if strategy.parameters is not None:
            # Save and evaluate global model
            print("Server completed. Using strategy parameters for final model.")
            global_model = save_global_model(strategy.parameters)
            evaluate_global_model(global_model, test_patients)
            
            # Save metrics history
            with open("results/server/metrics_history.json", "w") as f:
                json.dump(strategy.metrics_history, f, indent=2)
            
            # Plot learning curves
            plt.figure(figsize=(12, 6))
            rounds = list(range(1, len(strategy.metrics_history["train_reward"]) + 1))
            
            plt.subplot(1, 2, 1)
            plt.plot(rounds, strategy.metrics_history["train_reward"], label="Train")
            if strategy.metrics_history["val_reward"]:
                plt.plot(rounds, strategy.metrics_history["val_reward"], label="Validation")
            plt.xlabel("Round")
            plt.ylabel("Reward")
            plt.title("Federated Learning Rewards")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(rounds, strategy.metrics_history["train_loss"], label="Train")
            if strategy.metrics_history["val_loss"]:
                plt.plot(rounds, strategy.metrics_history["val_loss"], label="Validation")
            plt.xlabel("Round")
            plt.ylabel("Loss")
            plt.title("Federated Learning Loss")
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig("results/server/learning_curves.png")
            plt.close()
            
            return history, global_model
        else:
            print("Error: No parameters available in strategy after training.")
            return history, None
            
    except Exception as e:
        print(f"Error in server: {e}")
        traceback.print_exc()
        return None, None


# Fix for run_federated_simulation to handle model loading errors gracefully
def run_federated_simulation(num_rounds=10, num_clients=5, num_patients_per_client=5, server_address="127.0.0.1:8089"):
    """Run a full federated learning simulation with both server and clients"""
    print(f"Starting federated learning simulation with {num_clients} clients for {num_rounds} rounds")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/server", exist_ok=True)
    
    # Initialize clients and server in separate processes
    processes = []
    
    # Start server process
    server_process = mp.Process(
        target=start_server,
        args=(num_rounds, num_clients)
    )
    server_process.start()
    processes.append(server_process)
    
    # Wait for server to start
    time.sleep(2)
    
    # Start client processes
    for i in range(num_clients):
        client_process = mp.Process(
            target=start_client,
            args=(i, server_address, num_patients_per_client, 42 + i)  # Different seed for each client
        )
        client_process.start()
        processes.append(client_process)
    
    # Wait for all processes to complete
    for process in processes:
        process.join()
    
    print("Federated learning simulation completed!")
    
    # Load and evaluate global model - handle errors gracefully
    try:
        global_model = PPO(state_dim=6, action_dim=4, agent_id="global")
        global_model.load_model("results/server/global_model.pt")
        
        # Generate test patients
        test_patients = generate_test_patients(num_patients=10)
        
        # Evaluate global model
        evaluate_global_model(global_model, test_patients)
        
        # Compare with standalone agent if it exists
        try:
            standalone_agent = PPO(state_dim=6, action_dim=4, agent_id="standalone")
            standalone_agent.load_model("results/standalone/model.pt")
            
            # Compare trajectories
            env = gym.make("HIVPatient-v0")
            models = {
                "Global": global_model,
                "Standalone": standalone_agent
            }
            
            # Compare on a few patients
            for i, patient in enumerate(test_patients[:3]):
                compare_trajectories(models, env, patient["params"])
        except FileNotFoundError:
            print("Standalone model not found for comparison. Run standalone training first if you want to compare models.")
        
        return global_model
    except FileNotFoundError:
        print("Global model file not found. The federated learning process may have failed to save the model.")
        print("Check the server logs for details.")
        return None
    except Exception as e:
        print(f"Error loading or evaluating global model: {e}")
        traceback.print_exc()
        return None    
# Function to start a Flower client
def start_client(client_id, server_address="127.0.0.1:8089", num_patients=5, seed=None):
    """Start a Flower client with the specified parameters"""
    # Create client
    client = HIVClient(client_id, num_patients=num_patients, seed=seed)
    
    # Start client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )
    
    # Save model after federated learning
    client.save_model()
    
    return client

# Function to run federated learning simulation

# Main function
def main():
    """Main function to run the code"""
    parser = argparse.ArgumentParser(description="HIV Treatment Federated Learning")
    parser.add_argument("--mode", type=str, default="federated", choices=["federated", "client", "server", "standalone", "all"],
                      help="Mode to run: federated, client, server, standalone, or all")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--patients", type=int, default=5, help="Number of patients per client")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes for standalone training")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8089", help="Server address")
    parser.add_argument("--client_id", type=int, default=0, help="Client ID (only for client mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    if args.mode == "federated" or args.mode == "all":
        run_federated_simulation(
            num_rounds=args.rounds,
            num_clients=args.clients,
            num_patients_per_client=args.patients,
            server_address=args.server_address
        )
    
    if args.mode == "client" or args.mode == "all":
        start_client(
            client_id=args.client_id,
            server_address=args.server_address,
            num_patients=args.patients,
            seed=args.seed
        )
    
    if args.mode == "server" or args.mode == "all":
        start_server(
            num_rounds=args.rounds,
            num_clients=args.clients
        )
    
    if args.mode == "standalone" or args.mode == "all":
        env = gym.make("HIVPatient-v0")
        train_standalone_agent(
            env=env,
            num_episodes=args.episodes
        )
    
    print("Execution completed!")

if __name__ == "__main__":
    main()
