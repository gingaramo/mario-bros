import torch
import torch.nn as nn


def replace_linear_with_noisy(model: nn.Module):
  for name, module in model.named_children():
    if isinstance(module, nn.Linear):
      # Replace Linear with NoisyLinear
      in_features = module.in_features
      out_features = module.out_features
      noisy_module = NoisyLinear(in_features, out_features)
      setattr(model, name, noisy_module)
    else:
      # Recursively apply to submodules
      replace_linear_with_noisy(module)
  return model


class NoisyLinear(nn.Module):

  def __init__(self, in_features, out_features, std_init=0.4):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    # Initialize weights and biases
    self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
    self.bias = nn.Parameter(torch.Tensor(out_features))
    self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
    self.register_buffer('weight_epsilon',
                         torch.Tensor(out_features, in_features))
    self.register_buffer('bias_epsilon', torch.Tensor(out_features))

    self.reset_parameters(std_init)

  def reset_parameters(self, std_init):
    # Initialize weights and biases
    mu_range = 1 / self.in_features**0.5
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(std_init / self.in_features**0.5)
    self.bias.data.uniform_(-mu_range, mu_range)

  def forward(self, input):
    weight = self.weight_mu + self.weight_sigma * self.weight_epsilon.normal_()
    # Empirically bias noise is unhelpful, so we don't use it.
    # bias = self.bias + self.bias_sigma * self.bias_epsilon.normal_()
    return nn.functional.linear(input, weight, self.bias)
