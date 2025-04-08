import gymnasium as gym
import pygame
import torch
import pickle
import numpy as np

# Actor-Critic Network
class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()
        self.shared_layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )
        self.actor = torch.nn.Linear(64, output_dim)
        self.critic = torch.nn.Linear(64, 1)

    def forward(self, x):
        shared_features = self.shared_layers(x)
        policy = torch.nn.functional.softmax(self.actor(shared_features), dim=-1)
        value = self.critic(shared_features)
        return policy, value

# Load model function
def load_model(checkpoint_path):
    """Load model from a pickle file."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    model = ActorCriticNetwork(checkpoint['state_dim'], checkpoint['action_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    return model

class CartPoleRender:
    def __init__(self, model1, model2, fps=30):
        self.env1 = gym.make('CartPole-v1', render_mode='human')
        self.env2 = gym.make('CartPole-v1', render_mode='human')
        self.model1 = model1
        self.model2 = model2
        self.fps = fps

    def select_action(self, model, state):
        """Select action based on model's policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy, _ = model(state_tensor)
        action_probs = policy.detach().numpy()[0]
        action = np.random.choice(len(action_probs), p=action_probs)
        return action

    def render(self):
        """Render two environments simultaneously."""
        clock = pygame.time.Clock()

        # Reset environments
        state1, _ = self.env1.reset()
        state2, _ = self.env2.reset()

        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Select actions for both models
            action1 = self.select_action(self.model1, state1)
            action2 = self.select_action(self.model2, state2)

            # Step environments
            state1, _, done1, truncated1, _ = self.env1.step(action1)
            state2, _, done2, truncated2, _ = self.env2.step(action2)

            # Render environments
            self.env1.render()
            self.env2.render()

            # Reset environments if done
            if done1 or truncated1:
                state1, _ = self.env1.reset()
            if done2 or truncated2:
                state2, _ = self.env2.reset()

            clock.tick(self.fps)

        self.env1.close()
        self.env2.close()

import os

if __name__ == "__main__":
    input_dim = 4  # Số chiều của trạng thái (CartPole thường có 4)
    output_dim = 2  # Số hành động (CartPole có 2 hành động: trái/phải)

    # Load models
    model1 = load_model("Mountain Car/actor_critic_checkpoint.pkl", input_dim, output_dim)
    model2 = load_model("Mountain Car/ddqn_checkpoint.pkl", input_dim, output_dim)

    # Render environments
    renderer = CartPoleRender(model1, model2, fps=30)
    renderer.render()
