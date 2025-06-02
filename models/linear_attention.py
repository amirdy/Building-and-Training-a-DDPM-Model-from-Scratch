import torch
import torch.nn as nn

class LinearAttention(nn.Module):
  """ Linear Attention module with optional causal masking. """

  def __init__(self, embed_dim, num_heads=2):
      """ Initializes the Multi-Head Attention module.

      Args:
          embed_dim (int): Dimension of the input embeddings whic is the in_channels
          num_heads (int): Number of attention heads.
      """
      super().__init__()
      self.embed_dim = embed_dim
      self.W_query = nn.Linear(embed_dim, embed_dim, bias=False)
      self.W_key = nn.Linear(embed_dim, embed_dim, bias=False)
      self.W_value = nn.Linear(embed_dim, embed_dim, bias=False)
      self.out_proj = nn.Linear(embed_dim, embed_dim)  # Linear layer to combine head outputs
      self.elu = nn.ELU()
      self.num_heads = num_heads
      self.head_dim = embed_dim // num_heads
      assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
 
  def forward(self, input):
      """Computes multi-head attention.

      Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            Note that channels = embed_dim and height * width = seq_length
      Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width)
      """
      batch_size, c, h, w = input.shape      # Input shape: (batch_size, channels, height, width)
      seq_length = h * w
      embed_dim = c

      input = input.view(batch_size, c, seq_length).permute(0, 2, 1) # (batch_size, height*width, channels)
 
      K = self.W_key(input)    # (batch_size, seq_length, embed_dim)
      Q = self.W_query(input)  # (batch_size, seq_length, embed_dim)
      V = self.W_value(input)  # (batch_size, seq_length, embed_dim)

      K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)  # (batch_size, seq_length, num_heads, head_dim)
      Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)  # (batch_size, seq_length, num_heads, head_dim)
      V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)  # (batch_size, seq_length, num_heads, head_dim)

      K = K.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
      Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
      V = V.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)

      phi_K = self.elu(K) + 1
      phi_Q = self.elu(Q) + 1
      
      numerator = phi_Q @ ((phi_K.transpose(2,3)) @ V) 

      ones = torch.ones(batch_size,  self.num_heads, seq_length, 1).to(K.device)
      denominator = phi_Q @ (phi_K.transpose(2,3) @ ones)

      output = numerator / (denominator + 1e-6)

      output = output.permute(0, 2, 1, 3) # (batch_size, seq_length, num_heads, head_dim) 
      output = output.contiguous().view(batch_size, seq_length, self.embed_dim) # (batch_size, seq_length, embed_dim) 
      output = self.out_proj(output)  # (batch_size, seq_length, embed_dim) 
      
      output = output.permute(0, 2, 1)          # (batch_size, embed_dim, seq_length) 
      output = output.view(batch_size, c, h, w) # (batch_size, c, h, w) 
      

      return output