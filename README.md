# Quick-commerce-dynamic-pricing
# MAGC Quick Commerce Dynamic Pricing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch: 1.10+](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)
[![DGL: 0.9+](https://img.shields.io/badge/DGL-0.9+-green.svg)](https://www.dgl.ai/)

A PyTorch implementation of Multi-Agent Graph Convolutional Reinforcement Learning (MAGC) for dynamic pricing in quick commerce platforms. This project adapts the MAGC framework originally designed for EV charging stations to the quick commerce domain, optimizing pricing strategies for online retailers in real-time.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Experiments](#experiments)
- [Results](#results)
- [Extending the Model](#extending-the-model)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Overview

Quick commerce platforms in India frequently adjust product prices in real-time based on demand, competitor pricing, and customer behavior. This system uses a multi-agent reinforcement learning approach to optimize pricing strategies dynamically, maximizing profits while maintaining customer satisfaction.

The model leverages graph convolutional networks to capture the complex relationships between stores, including:
- Competition between stores of different retail chains
- Cooperation between stores of the same retail chain
- Spatial relationships and delivery zones
- Customer purchasing patterns and price sensitivity

### Motivation

Traditional pricing strategies rely on manual rules or simple automated adjustments that fail to capture the complex, interconnected nature of quick commerce markets. Our MAGC approach offers several advantages:

- **Adaptive Pricing**: Automatically adjusts prices based on time of day, demand patterns, and competitor behavior
- **Multi-Agent Learning**: Each store learns its own pricing strategy while considering other stores' actions
- **Network Effects**: Captures how pricing decisions propagate through the market
- **Balance**: Optimizes for short-term profit and long-term customer retention

## Key Features

- **Multi-Agent Reinforcement Learning**: Each store with dynamic pricing capability is modeled as an agent that learns optimal pricing strategies
- **Graph Convolutional Networks**: Capture competition and cooperation relationships between stores
- **Customer Behavior Modeling**: Simulates how customers respond to different pricing strategies based on price, delivery time, and inventory
- **Temporal Dynamics**: Learns time-dependent pricing patterns (e.g., different strategies for morning vs. evening)
- **Memory-Efficient Implementation**: Optimized to run on systems with limited resources (12GB RAM CPU)

## System Architecture

The system consists of the following key components:

### Environment

- **QuickCommerceEnv**: Simulates the quick commerce marketplace, including:
  - Stores with inventory and pricing information
  - Customer order generation based on time patterns
  - Customer decision model for store selection
  - Order fulfillment and reward calculation

### Agent Components

- **Actor Network**: Determines pricing actions for each store
  - Graph-based message passing between stores
  - GRU for temporal patterns
  - Separate processing for competition and cooperation

- **Critic Network**: Evaluates the value of pricing decisions
  - Provides feedback for policy improvement

- **Aggregator**: Combines features from multiple stores using attention
  - Separate attention for dynamic and fixed-price stores

- **ReplayBuffer**: Stores experiences for off-policy learning

### Training Pipeline

The training process follows these steps:
1. Generate synthetic data
2. Run baseline evaluation with fixed pricing
3. Train the MAGC agent over multiple epochs and days
4. Evaluate the trained agent against the baseline
5. Generate visualizations showing performance metrics

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- DGL (Deep Graph Library)

## Installation

### Option 1: Using pip

```bash
# Create a virtual environment
python -m venv magc-env
source magc-env/bin/activate  # On Windows: magc-env\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib dgl
```

### Option 2: Using conda

```bash
# Create a conda environment
conda create -n magc-env python=3.8
conda activate magc-env

# Install dependencies
conda install pytorch numpy matplotlib -c pytorch
pip install dgl
```

### Clone the Repository

```bash
git clone https://github.com/viznuv/magc-quick-commerce.git
cd magc-quick-commerce
```

## Usage

### Quick Start

Run the main script and follow the prompts:

```bash
python quick_commerce_magc_fixed.py
```

You'll be asked to select one of three modes:
1. **Quick test** (1 epoch, 3 days): Takes about 10-15 minutes
2. **Full training** (3 epochs, 5 days): Takes about 1-2 hours
3. **Evaluate pre-trained model** (if available)

### Custom Configuration

You can modify the `Args` class in the script to customize parameters:

```python
class Args:
    def __init__(self):
        # Environment parameters
        self.N = 20  # Number of stores
        self.n_zones = 40  # Number of zones
        self.n_chains = 5  # Number of store chains
        self.T_LEN = 24  # Time steps per day (hourly)
        
        # Model parameters
        self.price_scale = 2.0  # Price scaling factor
        self.hiddim = 32  # Hidden dimension
        
        # Training parameters
        self.epochs = 3  # Number of training epochs
        self.days = 5  # Number of days to simulate
        # ...
```

### Running Headless

For running on a server without interactive input:

```bash
# Quick test mode
python -c "import quick_commerce_magc_fixed; quick_commerce_magc_fixed.train_magc(n_epochs=1, n_days=3)"

# Full training mode
python -c "import quick_commerce_magc_fixed; quick_commerce_magc_fixed.train_magc(n_epochs=3, n_days=5)"
```

## Implementation Details

### Data Generation

The system generates synthetic data for:
- Store inventory levels
- Delivery times between zones and stores
- Base prices and costs with daily patterns
- Competitor prices
- Store chain relationships
- Distance matrices for spatial relationships

### Adjacency Matrices

Two types of adjacency matrices define store relationships:
1. **Competition Matrix**: Connects stores from different chains that are within a specified distance
2. **Cooperation Matrix**: Connects stores from the same chain that are within a specified distance

These matrices are used by the graph neural networks to propagate information between stores.

### Actor Network

The Actor network determines pricing actions for each store:

1. **Feature Processing**:
   - Processes store features like inventory, time, costs
   - Extracts time patterns using embeddings

2. **Graph Message Passing**:
   - Competition GNN: Captures influence from competing stores
   - Cooperation GNN: Captures influence from cooperating stores

3. **Temporal Processing**:
   - GRU cell maintains hidden state across time steps
   - Enables learning of time-dependent patterns

4. **Action Generation**:
   - Outputs pricing multipliers between 0 and 1
   - Scaled to actual price adjustments (0.7x to 1.3x base price)

### Customer Decision Model

The customer model simulates how customers choose stores based on:
1. **Price**: Lower prices increase selection probability
2. **Delivery Time**: Shorter delivery times are preferred
3. **Inventory**: Stores must have available inventory

The model uses a probabilistic selection mechanism where the probability of choosing a store is proportional to its score.

### Reward Function

Rewards have several components:
1. **Profit**: Primary reward based on (price - cost) * quantity
2. **Competitive Pricing**: Bonus for maintaining reasonable price ratios
3. **Customer Satisfaction**: Implicit in the customer selection model

### Training Process

The training follows a standard DDPG-like approach:
1. **Experience Collection**: Agent interacts with environment
2. **Replay Buffer**: Stores and samples transitions
3. **Critic Update**: Minimizes TD error
4. **Actor Update**: Maximizes expected reward
5. **Target Networks**: Soft updates for stability

## Experiments

### Baseline Comparison

The system compares the MAGC approach against a fixed-pricing baseline where all stores use their base prices without dynamic adjustment.

### Performance Metrics

Key metrics tracked during training and evaluation:
- **Total Profit**: Primary performance metric
- **Order Count**: Number of fulfilled orders
- **Average Price**: Average price across all fulfilled orders
- **Price Ratios**: How dynamic prices compare to base prices over time
- **Hourly Distribution**: How pricing strategies vary by hour of day

## Results

The MAGC approach typically achieves a **10-20% improvement in profit** compared to fixed pricing strategies. The most significant improvements are seen during:

1. **Peak Demand Periods**: Morning (8-12) and evening (17-21) hours
2. **High Competition Areas**: Zones with multiple competing stores
3. **Supply-Constrained Scenarios**: When inventory is limited

### Visualization

The system generates several visualizations:
- **Learning Curves**: Showing profit and reward improvements over time
- **Hourly Pricing Strategy**: How price adjustments vary by hour
- **Order Distribution**: Pattern of orders throughout the day

## Extending the Model

### Adding Real Data

To use real data instead of synthetic data:
1. Create a data loader module that reads from your data source
2. Format the data to match the expected structure
3. Replace the data generation code with your data loader

Example structure for order data:
```
{
  "zone_id": 123,
  "timestamp": "2023-05-18T14:30:00",
  "store_id": 456,
  "price": 599.99,
  "cost": 399.99,
  "quantity": 2
}
```

### Custom Customer Models

You can implement more sophisticated customer models by modifying the `step` method in `QuickCommerceEnv`:

```python
# Example: Price-elastic customer model
def price_elastic_customer(price, base_price, elasticity=1.5):
    return (base_price / price) ** elasticity
```

### Additional Features

Potential extensions to the model:
1. **Product Categories**: Different pricing strategies for different product types
2. **Weather Effects**: Incorporating weather data for demand prediction
3. **Promotional Events**: Special pricing during sales events
4. **Personalized Pricing**: Customer-specific pricing based on history

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

This work is adapted from:
```
@inproceedings{zhang2022multi,
  title={Multi-agent graph convolutional reinforcement learning for dynamic electric vehicle charging pricing},
  author={Zhang, Weijia and Liu, Hao and Han, Jindong and Ge, Yong and Xiong, Hui},
  booktitle={Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining},
  pages={2471--2481},
  year={2022}
}
```

## Contact

Your Name - vishnuprasad@sjmsom.in

Project Link: [https://github.com/viznuv/magc-quick-commerce](https://github.com/viznuv/magc-quick-commerce)
