import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import random
import json
import gc
from collections import deque

print("Setting up Quick Commerce MAGC system...")

# Create necessary directories
os.makedirs("./data", exist_ok=True)
os.makedirs("./params", exist_ok=True)
os.makedirs("./figures", exist_ok=True)

# Configuration parameters
class Args:
    def __init__(self):
        # System settings
        self.device = torch.device("cpu")
        self.seed = 42
        
        # Environment parameters
        self.N = 20  # Number of stores (reduced for CPU)
        self.n_zones = 40  # Number of zones
        self.n_chains = 5  # Number of store chains
        self.T_LEN = 24  # Time steps per day (hourly)
        self.interval = 1  # Hours per time step
        self.n_pred = 3  # Prediction window
        self.miss_time = 46  # Penalty time
        self.avg_purchase_qty = 4.5  # Average purchase quantity 
        self.std_purchase_qty = 2.0  # Std of purchase qty
        
        # Model parameters
        self.price_scale = 2.0  # Price scaling factor
        self.batch_size = 8  # Batch size (reduced)
        self.buffer_size = 500  # Replay buffer size (reduced)
        self.gamma = 0.99  # Discount factor
        self.hiddim = 32  # Hidden dimension (reduced)
        self.com_dim = 8  # Communication dimension (reduced)
        self.lr_a = 0.0001  # Actor learning rate
        self.lr_c = 0.0005  # Critic learning rate
        self.wdecay = 1e-5  # Weight decay
        self.dropout = 0.2  # Dropout rate
        self.eps = 1e-6  # Epsilon for numerical stability
        
        # Training parameters
        self.soft_tau_a = 0.01  # Soft update ratio for actor
        self.soft_tau_c = 0.01  # Soft update ratio for critic
        self.clip_norm = 1.0  # Gradient clip norm
        self.noise = True  # Use exploration noise
        self.std = 0.1  # Noise standard deviation
        self.epochs = 3  # Number of training epochs
        self.days = 5  # Number of days to simulate

args = Args()

# Set random seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# ReplayBuffer for storing transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.position = 0
        self.size = 0
    
    def push(self, transition):
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer[:self.size], batch_size)
        return zip(*batch)
    
    def __len__(self):
        return self.size

# Ornstein-Uhlenbeck noise for exploration
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.1):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.size) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

# Generate synthetic data
print("Generating synthetic data...")

# Store chains (representing different companies)
store_chains = np.random.randint(0, args.n_chains, size=args.N)

# Generate base product prices (daily cycle)
base_prices = np.random.normal(500, 100, size=(args.N, 24))
base_prices = np.clip(base_prices, 200, 800).astype(np.float32)

# Generate competitor prices (daily cycle)
competitor_prices = np.random.normal(500, 150, size=(args.N, 24))
competitor_prices = np.clip(competitor_prices, 180, 900).astype(np.float32)

# Generate product costs (daily cycle)
product_costs = base_prices * np.random.uniform(0.5, 0.7, size=(args.N, 24))
product_costs = np.clip(product_costs, 100, 600).astype(np.float32)

# Delivery times from zones to stores (in minutes)
delivery_times = np.random.normal(15, 10, size=(args.n_zones, args.N))
delivery_times = np.clip(delivery_times, 1, 60).astype(np.int32)

# Distances from zones to stores (in meters)
distances = np.random.normal(4000, 2000, size=(args.n_zones, args.N))
distances = np.clip(distances, 500, 15000).astype(np.int32)

# Store zones (surrounding areas for each store)
store_zones = []
for i in range(args.N):
    n_nearby = np.random.randint(5, 16)
    nearby_zones = np.random.choice(args.n_zones, size=n_nearby, replace=False)
    store_zones.append(nearby_zones.tolist())

# Distance matrix between stores
distance_matrix = np.random.normal(5000, 3000, size=(args.N, args.N))
distance_matrix = np.clip(distance_matrix, 0, 20000).astype(np.float32)
distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Make symmetric
np.fill_diagonal(distance_matrix, 0)  # Zero diagonal

# Choose stores with dynamic pricing capability
n_dp_stores = int(args.N * 0.6)  # 60% of stores can use dynamic pricing
dp_store_id = np.random.choice(args.N, size=n_dp_stores, replace=False).tolist()
fp_store_id = list(set(range(args.N)) - set(dp_store_id))

print(f"Number of stores: {args.N}")
print(f"Number of dynamic pricing stores: {n_dp_stores}")
print(f"Number of fixed pricing stores: {args.N - n_dp_stores}")

# Create adjacency matrices
adj_matrix_comp = np.zeros((args.N, args.N))
adj_matrix_coop = np.zeros((args.N, args.N))

for i in range(args.N):
    for j in range(args.N):
        if i != j and distance_matrix[i, j] < 8000:  # Within 8km
            if store_chains[i] != store_chains[j]:
                adj_matrix_comp[i, j] = 1  # Competition
            else:
                adj_matrix_coop[i, j] = 1  # Cooperation

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_stores, dp_store_ids):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_stores = n_stores
        self.dp_store_ids = dp_store_ids
        
        # Processing store features
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Store interactions via graph structure
        self.comp_fc = nn.Linear(hidden_dim, hidden_dim)
        self.coop_fc = nn.Linear(hidden_dim, hidden_dim)
        
        # Price policy output
        self.fc3 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        
        # Recurrent component for temporal patterns
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self, state, adj_comp, adj_coop, hidden=None):
        """
        Forward pass through the actor network
        
        Args:
            state: Store state tensor (batch_size, n_stores, state_dim)
            adj_comp: Competition adjacency matrix (batch_size, n_stores, n_stores) or (n_stores, n_stores)
            adj_coop: Cooperation adjacency matrix (batch_size, n_stores, n_stores) or (n_stores, n_stores)
            hidden: Previous hidden state (optional)
            
        Returns:
            actions: Price adjustments (batch_size, n_stores, 1)
            h: New hidden state
        """
        batch_size = state.size(0)
        
        # Process store features
        x = self.relu(self.fc1(state))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Make sure adjacency matrices have proper dimensions
        if adj_comp.dim() == 2:
            batch_adj_comp = adj_comp.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            batch_adj_comp = adj_comp
            
        if adj_coop.dim() == 2:
            batch_adj_coop = adj_coop.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            batch_adj_coop = adj_coop
        
        # Process competition influence
        x_comp = torch.bmm(batch_adj_comp, x)
        x_comp = self.relu(self.comp_fc(x_comp))
        
        # Process cooperation influence
        x_coop = torch.bmm(batch_adj_coop, x)
        x_coop = self.relu(self.coop_fc(x_coop))
        
        # Combine all information
        combined = torch.cat([x, x_comp, x_coop], dim=2)
        
        # Final processing
        x = self.relu(self.fc3(combined))
        x = self.dropout(x)
        
        # Update hidden state if provided
        if hidden is not None:
            h_flat = hidden.view(-1, self.hidden_dim)
            x_flat = x.view(-1, self.hidden_dim)
            h_new = self.gru(x_flat, h_flat)
            h = h_new.view(batch_size, self.n_stores, self.hidden_dim)
        else:
            h = x.clone()
        
        # Output price adjustments
        actions = self.sigmoid(self.out(x))  # Values between 0 and 1
        
        return actions, h

# Critic Network
class Critic(nn.Module):
    def __init__(self, hidden_dim):
        super(Critic, self).__init__()
        
        # Process store features
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Process aggregated features
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(args.dropout)
    
    def forward(self, x):
        """
        Forward pass through the critic network
        
        Args:
            x: Aggregated state-action representation (batch_size, hidden_dim)
            
        Returns:
            q: Q-value (batch_size, 1)
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        q = self.out(x)
        return q

# Store feature aggregator
class Aggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_stores, dp_store_ids):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_stores = n_stores
        self.dp_store_ids = dp_store_ids
        
        # Attention for dynamic pricing stores
        self.attn_dp = nn.Linear(input_dim, 1)
        
        # Attention for fixed pricing stores
        self.attn_fp = nn.Linear(input_dim, 1)
        
        # Process after aggregation
        self.fc = nn.Linear(input_dim * 4, hidden_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, dp_mask):
        """
        Aggregate store features with attention
        
        Args:
            x: Store features (batch_size, n_stores, input_dim)
            dp_mask: Dynamic pricing store mask (batch_size, n_stores, 1)
            
        Returns:
            agg: Aggregated representation (batch_size, hidden_dim)
        """
        # Compute attention scores
        attn_scores_dp = self.attn_dp(x)
        attn_scores_fp = self.attn_fp(x)
        
        # Mask scores based on store type
        attn_scores_dp = torch.where(dp_mask > 0.5, attn_scores_dp, torch.tensor(-1e9, device=x.device))
        attn_scores_fp = torch.where(dp_mask < 0.5, attn_scores_fp, torch.tensor(-1e9, device=x.device))
        
        # Apply softmax to get attention weights
        attn_weights_dp = F.softmax(attn_scores_dp, dim=1)
        attn_weights_fp = F.softmax(attn_scores_fp, dim=1)
        
        # Weighted sum for each group
        dp_sum = torch.sum(x * attn_weights_dp, dim=1)
        fp_sum = torch.sum(x * attn_weights_fp, dim=1)
        
        # Max pooling for each group
        dp_max, _ = torch.max(x * (dp_mask > 0.5).float(), dim=1)
        fp_max, _ = torch.max(x * (dp_mask < 0.5).float(), dim=1)
        
        # Combine all aggregated features
        combined = torch.cat([dp_sum, dp_max, fp_sum, fp_max], dim=1)
        agg = self.relu(self.fc(combined))
        
        return agg

# Environment for Quick Commerce simulation
class QuickCommerceEnv:
    def __init__(self):
        self.n_stores = args.N
        self.n_zones = args.n_zones
        self.T_LEN = args.T_LEN
        
        # Store data
        self.store_chains = store_chains
        self.base_prices = base_prices
        self.competitor_prices = competitor_prices
        self.product_costs = product_costs
        self.delivery_times = delivery_times
        self.distances = distances
        self.dp_store_id = dp_store_id
        
        # Initialize state
        self.current_day = 0
        self.current_hour = 0
        self.inventory = None
        self.pending_orders = None
        self.current_prices = None
        self.orders = None
        self.fulfilled_orders = None
        
    def reset(self, day=0):
        """Reset environment to beginning of a day"""
        self.current_day = day
        self.current_hour = 0
        
        # Initialize inventory (some randomness per day)
        self.inventory = np.random.normal(100, 20, size=(self.n_stores,))
        self.inventory = np.clip(self.inventory, 20, 200).astype(np.int32)
        
        # Reset prices to base prices
        self.current_prices = self.base_prices.copy()
        
        # Reset order tracking
        self.pending_orders = np.zeros((self.n_stores,))
        self.orders = [[] for _ in range(self.T_LEN)]
        self.fulfilled_orders = []
        
        # Generate orders for the day
        self._generate_orders()
        
        return self._get_state()
    
    def _generate_orders(self):
        """Generate orders for the day based on time patterns"""
        order_id = 0
        
        for hour in range(self.T_LEN):
            # Number of orders depends on time of day
            if 8 <= hour < 12:  # Morning peak
                n_orders = np.random.poisson(8)
            elif 17 <= hour < 21:  # Evening peak  
                n_orders = np.random.poisson(10)
            else:  # Off-peak
                n_orders = np.random.poisson(4)
                
            # Generate each order
            for _ in range(n_orders):
                zone = np.random.randint(0, self.n_zones)
                self.orders[hour].append((zone, hour, order_id))
                order_id += 1
    
    def _get_state(self):
        """Get the current state of the environment"""
        state = np.zeros((self.n_stores, 6))
        
        # Time feature
        state[:, 0] = self.current_hour
        
        # Store chain
        state[:, 1] = self.store_chains
        
        # Inventory
        state[:, 2] = self.inventory / 100.0  # Normalize
        
        # Current prices
        state[:, 3] = self.current_prices[:, self.current_hour] / 500.0  # Normalize
        
        # Product costs
        state[:, 4] = self.product_costs[:, self.current_hour] / 500.0  # Normalize
        
        # Pending orders
        state[:, 5] = self.pending_orders / 10.0  # Normalize
        
        return state
    
    def step(self, actions):
        """
        Take a step in the environment with the given pricing actions
        
        Args:
            actions: Price adjustment actions (n_dp_stores,)
            
        Returns:
            next_state: Next state after taking action
            rewards: Rewards for each store with dynamic pricing
            done: Whether episode is done
        """
        # Apply price adjustments to dynamic pricing stores
        for i, store_id in enumerate(self.dp_store_id):
            # Scale action to price adjustment (0.7x to 1.3x base price)
            price_multiplier = 0.7 + actions[i] * 0.6
            self.current_prices[store_id, self.current_hour] = (
                price_multiplier * self.base_prices[store_id, self.current_hour]
            )
        
        # Process current hour's orders
        current_orders = self.orders[self.current_hour]
        rewards = np.zeros(len(self.dp_store_id))
        
        for zone, _, order_id in current_orders:
            # Compute customer preference scores for each store
            store_scores = np.zeros(self.n_stores)
            
            for s in range(self.n_stores):
                # Price factor (lower prices are better)
                price = self.current_prices[s, self.current_hour]
                base = self.base_prices[s, self.current_hour]
                price_factor = np.exp(-0.5 * (price / base - 1.0) ** 2)
                
                # Delivery time factor (shorter times are better)
                time = self.delivery_times[zone, s]
                time_factor = np.exp(-time / 30.0)
                
                # Inventory factor (having inventory is required)
                inventory_factor = 1.0 if self.inventory[s] > 0 else 0.01
                
                # Combine factors
                store_scores[s] = price_factor * time_factor * inventory_factor
            
            # Customer chooses a store
            if np.sum(store_scores) > 0:
                probs = store_scores / np.sum(store_scores)
                chosen_store = np.random.choice(self.n_stores, p=probs)
                
                # Process order if inventory available
                if self.inventory[chosen_store] > 0:
                    # Reduce inventory
                    self.inventory[chosen_store] -= 1
                    
                    # Calculate purchase amount
                    purchase_qty = max(1, np.random.normal(args.avg_purchase_qty, args.std_purchase_qty))
                    
                    # Calculate profit
                    price = self.current_prices[chosen_store, self.current_hour]
                    cost = self.product_costs[chosen_store, self.current_hour]
                    profit = (price - cost) * purchase_qty
                    
                    # Record fulfilled order
                    self.fulfilled_orders.append({
                        'zone': zone,
                        'store': chosen_store,
                        'hour': self.current_hour,
                        'price': price,
                        'cost': cost,
                        'qty': purchase_qty,
                        'profit': profit
                    })
                    
                    # Assign reward to the store if it's dynamic pricing
                    if chosen_store in self.dp_store_id:
                        idx = self.dp_store_id.index(chosen_store)
                        rewards[idx] += profit / 500.0  # Scale rewards
                        
                        # Additional reward for competitive pricing
                        base_price = self.base_prices[chosen_store, self.current_hour]
                        price_ratio = price / base_price
                        if 0.85 <= price_ratio <= 1.05:
                            # Bonus for reasonable pricing
                            rewards[idx] += 0.05
        
        # Move to next hour
        self.current_hour += 1
        done = (self.current_hour >= self.T_LEN)
        
        # Get next state
        next_state = self._get_state() if not done else None
        
        return next_state, rewards, done
    
    def get_metrics(self):
        """Get performance metrics for the current episode"""
        if not self.fulfilled_orders:
            return {
                'total_orders': 0,
                'total_profit': 0,
                'dp_orders': 0,
                'dp_profit': 0,
                'avg_price': 0,
                'hourly_profits': {}
            }
            
        # Total metrics
        total_orders = len(self.fulfilled_orders)
        total_profit = sum(order['profit'] for order in self.fulfilled_orders)
        
        # Dynamic pricing store metrics
        dp_orders = sum(1 for order in self.fulfilled_orders if order['store'] in self.dp_store_id)
        dp_profit = sum(order['profit'] for order in self.fulfilled_orders if order['store'] in self.dp_store_id)
        
        # Average price
        avg_price = np.mean([order['price'] for order in self.fulfilled_orders])
        
        # Hourly metrics
        hourly_profits = {}
        for order in self.fulfilled_orders:
            hour = order['hour']
            hourly_profits[hour] = hourly_profits.get(hour, 0) + order['profit']
        
        return {
            'total_orders': total_orders,
            'total_profit': total_profit,
            'dp_orders': dp_orders,
            'dp_profit': dp_profit,
            'avg_price': avg_price,
            'hourly_profits': hourly_profits
        }

# MAGC Agent
class MAGCAgent:
    def __init__(self):
        self.state_dim = 6
        self.hidden_dim = args.hiddim
        self.n_stores = args.N
        self.n_dp_stores = len(dp_store_id)
        
        # Prepare torch tensors of adjacency matrices
        self.adj_comp = torch.FloatTensor(adj_matrix_comp)
        self.adj_coop = torch.FloatTensor(adj_matrix_coop)
        
        # Create dynamic pricing store mask
        self.dp_mask = torch.zeros(1, self.n_stores, 1)
        for idx in dp_store_id:
            self.dp_mask[0, idx, 0] = 1.0
        
        # Initialize networks
        self.actor = Actor(self.state_dim, self.hidden_dim, self.n_stores, dp_store_id)
        self.actor_target = Actor(self.state_dim, self.hidden_dim, self.n_stores, dp_store_id)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.hidden_dim)
        self.critic_target = Critic(self.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.aggregator = Aggregator(self.hidden_dim, self.hidden_dim, self.n_stores, dp_store_id)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_a, weight_decay=args.wdecay)
        self.critic_optimizer = optim.Adam(
            list(self.critic.parameters()) + list(self.aggregator.parameters()),
            lr=args.lr_c, 
            weight_decay=args.wdecay
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        
        # Exploration noise
        self.noise = OUNoise(self.n_dp_stores)
        
        # Recurrent hidden state
        self.hidden = None
    
    def select_action(self, state, add_noise=True):
        """
        Select pricing actions based on current state
        
        Args:
            state: Current environment state (n_stores, state_dim)
            add_noise: Whether to add exploration noise
            
        Returns:
            actions: Price adjustment actions for dynamic pricing stores
        """
        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get actions from actor network
        with torch.no_grad():
            actions, self.hidden = self.actor(
                state_tensor, 
                self.adj_comp, 
                self.adj_coop,
                self.hidden
            )
            actions = actions.squeeze(0).numpy()  # Remove batch dimension
        
        # Extract actions for dynamic pricing stores
        dp_actions = actions[dp_store_id].squeeze()
        
        # Add exploration noise if in training
        if add_noise:
            noise = self.noise.sample()
            dp_actions += noise
            dp_actions = np.clip(dp_actions, 0.0, 1.0)
        
        return dp_actions
    
    def update(self, batch_size):
        """
        Update policy and value parameters using sampled batch from replay buffer
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            critic_loss: Loss value from critic update
            actor_loss: Loss value from actor update
        """
        if len(self.replay_buffer) < batch_size:
            return 0, 0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        # Prepare batch adjacency matrices
        batch_adj_comp = self.adj_comp.unsqueeze(0).expand(batch_size, -1, -1)
        batch_adj_coop = self.adj_coop.unsqueeze(0).expand(batch_size, -1, -1)
        batch_dp_mask = self.dp_mask.expand(batch_size, -1, -1)
        
        # ------ Update critic ------
        self.critic_optimizer.zero_grad()
        
        # Get current Q values
        with torch.no_grad():
            curr_features, curr_hidden = self.actor(states, batch_adj_comp, batch_adj_coop)
        
        # Aggregate features
        curr_agg = self.aggregator(curr_hidden, batch_dp_mask)
        curr_q = self.critic(curr_agg)
        
        # Get next Q values
        with torch.no_grad():
            next_features, next_hidden = self.actor_target(next_states, batch_adj_comp, batch_adj_coop)
            next_agg = self.aggregator(next_hidden, batch_dp_mask)
            next_q = self.critic_target(next_agg)
            target_q = rewards + (1 - dones) * args.gamma * next_q
        
        # Compute critic loss
        critic_loss = F.mse_loss(curr_q, target_q)
        
        # Update critic
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.aggregator.parameters()), 
            args.clip_norm
        )
        self.critic_optimizer.step()
        
        # ------ Update actor ------
        self.actor_optimizer.zero_grad()
        
        # Get actor loss
        features, hidden = self.actor(states, batch_adj_comp, batch_adj_coop)
        agg = self.aggregator(hidden, batch_dp_mask)
        actor_loss = -self.critic(agg).mean()
        
        # Update actor
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), args.clip_norm)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor_target, self.actor, args.soft_tau_a)
        self._soft_update(self.critic_target, self.critic, args.soft_tau_c)
        
        return critic_loss.item(), actor_loss.item()
        
    def _soft_update(self, target, source, tau):
        """Soft update target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def save(self, filepath):
        """Save model parameters"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'aggregator': self.aggregator.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, filepath)
    
    def load(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.aggregator.load_state_dict(checkpoint['aggregator'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

# Training function
def train_magc(n_epochs=None, n_days=None):
    """Train the MAGC agent"""
    if n_epochs is None:
        n_epochs = args.epochs
    if n_days is None:
        n_days = args.days
        
    print("\nTraining MAGC for Quick Commerce Pricing...")
    print(f"Settings: {n_epochs} epochs, {n_days} days per epoch")
    
    # Initialize environment and agent
    env = QuickCommerceEnv()
    agent = MAGCAgent()
    
    # Tracking metrics
    all_rewards = []
    all_profits = []
    all_prices = []
    critic_losses = []
    actor_losses = []
    
    # Run baseline evaluation for comparison
    baseline_profits = []
    print("\nRunning baseline (fixed pricing) for comparison...")
    
    for day in range(min(n_days, 3)):
        env_baseline = QuickCommerceEnv()
        state = env_baseline.reset(day)
        done = False
        
        while not done:
            # Use constant action (keeps base price)
            actions = np.ones(len(dp_store_id)) * 0.5
            next_state, _, done = env_baseline.step(actions)
            state = next_state
        
        metrics = env_baseline.get_metrics()
        baseline_profits.append(metrics['total_profit'])
        print(f"Baseline Day {day+1}: Orders: {metrics['total_orders']}, Profit: ${metrics['total_profit']:.2f}")
    
    baseline_avg_profit = np.mean(baseline_profits)
    print(f"Baseline Average Daily Profit: ${baseline_avg_profit:.2f}")
    
    # Memory cleanup
    del env_baseline
    gc.collect()
    
    # Start training
    start_time = time.time()
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        epoch_rewards = []
        epoch_profits = []
        epoch_prices = []
        
        for day in range(n_days):
            print(f"  Day {day+1}/{n_days}... ", end="", flush=True)
            
            # Reset environment and agent for new day
            state = env.reset(day % n_days)
            agent.noise.reset()
            agent.hidden = None
            
            day_reward = 0
            day_steps = 0
            
            # Run episode
            done = False
            while not done:
                # Select action
                action = agent.select_action(state, add_noise=True)
                
                # Take action in environment
                next_state, reward, done = env.step(action)
                
                # Store transition in replay buffer
                if not done:
                    agent.replay_buffer.push((state, action, reward, next_state, 0))
                else:
                    agent.replay_buffer.push((state, action, reward, state, 1))
                
                # Train agent
                critic_loss, actor_loss = agent.update(args.batch_size)
                
                # Record losses if valid
                if critic_loss > 0:
                    critic_losses.append(critic_loss)
                if actor_loss > 0:
                    actor_losses.append(actor_loss)
                
                # Update for next step
                state = next_state
                day_reward += np.sum(reward)
                day_steps += 1
                
                # Free up memory periodically
                if day_steps % 12 == 0:
                    gc.collect()
            
            # Get daily metrics
            metrics = env.get_metrics()
            epoch_rewards.append(day_reward)
            epoch_profits.append(metrics['total_profit'])
            epoch_prices.append(metrics['avg_price'])
            
            # Print progress
            print(f"Done. Orders: {metrics['total_orders']}, Profit: ${metrics['total_profit']:.2f}, Reward: {day_reward:.2f}")
        
        # Record epoch metrics
        all_rewards.extend(epoch_rewards)
        all_profits.extend(epoch_profits)
        all_prices.extend(epoch_prices)
        
        # Calculate epoch statistics
        epoch_avg_profit = np.mean(epoch_profits)
        epoch_avg_price = np.mean([p for p in epoch_prices if p > 0])
        epoch_avg_reward = np.mean(epoch_rewards)
        
        # Compare with baseline
        improvement = (epoch_avg_profit - baseline_avg_profit) / baseline_avg_profit * 100
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Average Profit: ${epoch_avg_profit:.2f} ({improvement:.1f}% vs baseline)")
        print(f"  Average Price: ${epoch_avg_price:.2f}")
        print(f"  Average Reward: {epoch_avg_reward:.2f}")
        
        # Save model after each epoch
        agent.save(f"./params/magc_quick_commerce_epoch_{epoch+1}.pt")
        
        # Plot learning curves periodically
        if (epoch + 1) % max(1, n_epochs // 2) == 0:
            # Plot profits
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(all_rewards)
            plt.title('Rewards')
            plt.xlabel('Day')
            plt.ylabel('Total Reward')
            
            plt.subplot(1, 3, 2)
            plt.plot(all_profits)
            plt.axhline(y=baseline_avg_profit, color='r', linestyle='--', label='Baseline')
            plt.title('Daily Profits')
            plt.xlabel('Day')
            plt.ylabel('Profit ($)')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            plt.plot(all_prices)
            plt.title('Average Prices')
            plt.xlabel('Day')
            plt.ylabel('Price ($)')
            
            plt.tight_layout()
            plt.savefig(f'./figures/learning_curves_epoch_{epoch+1}.png')
            plt.close()
    
    # Training complete
    training_time = time.time() - start_time
    
    print("\nTraining complete!")
    print(f"Total training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    # Save final model
    agent.save("./params/magc_quick_commerce_final.pt")
    
    # Plot final learning curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(all_rewards)
    plt.title('Rewards per Day')
    plt.xlabel('Day')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(all_profits)
    plt.axhline(y=baseline_avg_profit, color='r', linestyle='--', label='Baseline')
    plt.title('Profits per Day')
    plt.xlabel('Day')
    plt.ylabel('Profit ($)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(all_prices)
    plt.title('Average Prices per Day')
    plt.xlabel('Day')
    plt.ylabel('Price ($)')
    
    if critic_losses and actor_losses:
        plt.subplot(2, 2, 4)
        plt.plot(critic_losses, label='Critic Loss')
        plt.plot(actor_losses, label='Actor Loss')
        plt.title('Training Losses')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('./figures/final_learning_curves.png')
    plt.close()
    
    return agent, env, (all_rewards, all_profits, all_prices)

# Evaluation function
def evaluate_magc(agent=None, n_days=5):
    """Evaluate a trained MAGC agent"""
    print("\nEvaluating MAGC for Quick Commerce Pricing...")
    
    # Load agent if not provided
    if agent is None:
        agent = MAGCAgent()
        try:
            agent.load("./params/magc_quick_commerce_final.pt")
            print("Loaded model from ./params/magc_quick_commerce_final.pt")
        except:
            print("No saved model found. Training a new model...")
            agent, _, _ = train_magc(n_epochs=1, n_days=3)
    
    # Initialize environment
    env = QuickCommerceEnv()
    
    # Run evaluation
    daily_metrics = []
    hourly_pricing = {h: [] for h in range(24)}
    
    for day in range(n_days):
        print(f"\nEvaluating Day {day+1}/{n_days}...")
        
        # Reset environment and agent
        state = env.reset(day)
        agent.hidden = None
        done = False
        
        while not done:
            # Select action without exploration noise
            action = agent.select_action(state, add_noise=False)
            
            # Log hourly pricing decisions
            hour = int(state[0, 0])
            for i, store_id in enumerate(dp_store_id):
                # Calculate price ratio (current / base)
                current_price = env.current_prices[store_id, hour]
                base_price = env.base_prices[store_id, hour]
                price_ratio = current_price / base_price
                hourly_pricing[hour].append(price_ratio)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            state = next_state
        
        # Get day metrics
        metrics = env.get_metrics()
        daily_metrics.append(metrics)
        
        print(f"  Results: Orders: {metrics['total_orders']}, Profit: ${metrics['total_profit']:.2f}")
        print(f"  Dynamic Pricing Orders: {metrics['dp_orders']}, Average Price: ${metrics['avg_price']:.2f}")
    
    # Calculate overall metrics
    total_orders = sum(m['total_orders'] for m in daily_metrics)
    total_profit = sum(m['total_profit'] for m in daily_metrics)
    dp_orders = sum(m['dp_orders'] for m in daily_metrics)
    avg_price = np.mean([m['avg_price'] for m in daily_metrics if m['avg_price'] > 0])
    
    print("\nEvaluation Summary:")
    print(f"Total Orders: {total_orders} ({total_orders/n_days:.1f} per day)")
    print(f"Total Profit: ${total_profit:.2f} (${total_profit/n_days:.2f} per day)")
    print(f"Dynamic Pricing Orders: {dp_orders} ({dp_orders/n_days:.1f} per day)")
    print(f"Average Price: ${avg_price:.2f}")
    
    # Run baseline comparison
    baseline_profits = []
    baseline_orders = []
    
    for day in range(n_days):
        env_baseline = QuickCommerceEnv()
        state = env_baseline.reset(day)
        done = False
        
        while not done:
            # Use base prices (no dynamic adjustment)
            actions = np.ones(len(dp_store_id)) * 0.5
            next_state, _, done = env_baseline.step(actions)
            state = next_state
        
        metrics = env_baseline.get_metrics()
        baseline_profits.append(metrics['total_profit'])
        baseline_orders.append(metrics['total_orders'])
    
    baseline_total_profit = sum(baseline_profits)
    baseline_total_orders = sum(baseline_orders)
    
    profit_improvement = (total_profit - baseline_total_profit) / baseline_total_profit * 100
    order_improvement = (total_orders - baseline_total_orders) / baseline_total_orders * 100
    
    print("\nComparison with Fixed Pricing Baseline:")
    print(f"Profit: ${total_profit:.2f} vs ${baseline_total_profit:.2f} (+{profit_improvement:.1f}%)")
    print(f"Orders: {total_orders} vs {baseline_total_orders} (+{order_improvement:.1f}%)")
    
    # Plot hourly pricing patterns
    plt.figure(figsize=(15, 6))
    
    # Calculate average price ratio by hour
    hourly_avg_ratio = [np.mean(hourly_pricing[h]) if hourly_pricing[h] else 1.0 for h in range(24)]
    hourly_std_ratio = [np.std(hourly_pricing[h]) if len(hourly_pricing[h]) > 1 else 0.0 for h in range(24)]
    
    # Plot with error bars
    plt.errorbar(range(24), hourly_avg_ratio, yerr=hourly_std_ratio, marker='o', capsize=4)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Base Price')
    plt.title('Dynamic Pricing Strategy by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Price Ratio (Relative to Base Price)')
    plt.xticks(range(0, 24, 2))
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./figures/hourly_pricing_strategy.png')
    plt.close()
    
    # Plot hourly order distribution
    hourly_orders = {}
    for metrics in daily_metrics:
        for hour, profit in metrics['hourly_profits'].items():
            hourly_orders[hour] = hourly_orders.get(hour, 0) + 1
    
    # Fill in missing hours
    for hour in range(24):
        if hour not in hourly_orders:
            hourly_orders[hour] = 0
    
    # Sort by hour
    hourly_order_data = [(hour, count) for hour, count in sorted(hourly_orders.items())]
    hours, counts = zip(*hourly_order_data)
    
    plt.figure(figsize=(15, 6))
    plt.bar(hours, counts)
    plt.title('Order Distribution by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Orders')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('./figures/hourly_order_distribution.png')
    plt.close()
    
    return daily_metrics, hourly_pricing

# Main function to run the whole system
def main():
    print("Quick Commerce Dynamic Pricing with MAGC")
    print("======================================")
    
    # Ask for mode
    print("\nSelect mode:")
    print("1. Quick test (1 epoch, 3 days)")
    print("2. Full training (3 epochs, 5 days)")
    print("3. Evaluate pre-trained model")
    
    mode = input("Enter mode (1-3): ").strip()
    
    if mode == '1':
        # Quick test
        agent, env, metrics = train_magc(n_epochs=1, n_days=3)
        evaluate_magc(agent, n_days=2)
    elif mode == '2':
        # Full training
        agent, env, metrics = train_magc(n_epochs=3, n_days=5)
        evaluate_magc(agent, n_days=3)
    elif mode == '3':
        # Evaluation only
        try:
            evaluate_magc(n_days=3)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training a quick model instead...")
            agent, env, metrics = train_magc(n_epochs=1, n_days=3)
            evaluate_magc(agent, n_days=2)
    else:
        print("Invalid mode. Running quick test...")
        agent, env, metrics = train_magc(n_epochs=1, n_days=3)
        evaluate_magc(agent, n_days=2)
    
    print("\nDone! Check the ./figures directory for visualizations.")

if __name__ == "__main__":
    main()