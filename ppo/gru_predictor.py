import torch
import torch.nn as nn
import torch.nn.functional as F
from ppo.alg_parameters import NetParameters


class GRUPredictor(nn.Module):
    """
    GRU-based predictor for target position when occluded by obstacles.

    Input: Sequence of (relative_x, relative_y, visibility_flag)
    Output: Predicted (relative_x, relative_y) for current timestep

    Args:
        seq_len: Number of timesteps to look back
        hidden_dim: GRU hidden dimension
        num_layers: Number of GRU layers
    """
    def __init__(self,
                 seq_len=NetParameters.CONTEXT_LEN,
                 input_dim=3,
                 hidden_dim=64,
                 num_layers=2,
                 output_dim=2):
        super(GRUPredictor, self).__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # Output projection layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)

    def forward(self, seq_input, hidden=None):
        """
        Forward pass.

        Args:
            seq_input: Tensor of shape (batch_size, seq_len, input_dim)
                      Contains [relative_x, relative_y, visibility_flag] for each timestep
            hidden: Initial hidden state (optional)

        Returns:
            predictions: Tensor of shape (batch_size, output_dim)
            hidden: Final hidden state
        """
        batch_size = seq_input.size(0)

        # GRU forward pass
        gru_out, hidden = self.gru(seq_input, hidden)

        # Use the last output for prediction
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_dim)

        # Project to output
        predictions = self.fc(last_output)  # (batch_size, output_dim)

        # Apply sigmoid to ensure predictions are in [0, 1]
        predictions = torch.sigmoid(predictions)

        return predictions, hidden

    def get_loss(self, predictions, targets):
        """
        Compute loss between predictions and ground truth targets.

        Args:
            predictions: Tensor of shape (batch_size, 2)
            targets: Tensor of shape (batch_size, 2) with normalized [0, 1] coordinates

        Returns:
            loss: Scalar tensor
        """
        return F.mse_loss(predictions, targets)


class GRUPredictorWrapper:
    """
    Wrapper for managing GRU predictor with sequence buffer.
    """
    def __init__(self, model, device, seq_len=None):
        self.model = model
        self.device = device
        self.seq_len = seq_len or model.seq_len

        # Sequence buffer: list of (relative_x, relative_y, visibility_flag)
        self.sequence_buffer = []

    def reset(self):
        """Reset sequence buffer."""
        self.sequence_buffer = []

    def add_observation(self, rel_x, rel_y, is_visible):
        """
        Add observation to sequence buffer.

        Args:
            rel_x: Relative x coordinate (normalized to [0, 1])
            rel_y: Relative y coordinate (normalized to [0, 1])
            is_visible: Boolean indicating if target is visible
        """
        flag = 1.0 if is_visible else 0.0

        # If not visible, set position to -1
        if not is_visible:
            rel_x = -1.0
            rel_y = -1.0

        self.sequence_buffer.append([rel_x, rel_y, flag])

        # Keep only the last seq_len observations
        if len(self.sequence_buffer) > self.seq_len:
            self.sequence_buffer = self.sequence_buffer[-self.seq_len:]

    def predict(self):
        """
        Make prediction based on current sequence buffer.

        Returns:
            prediction: Array of [pred_x, pred_y] or None if buffer is empty
        """
        if len(self.sequence_buffer) == 0:
            return None

        # Prepare input tensor
        seq_input = torch.tensor(
            self.sequence_buffer,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # (1, seq_len, 3)

        # Pad if sequence is shorter than seq_len
        if len(self.sequence_buffer) < self.seq_len:
            pad_len = self.seq_len - len(self.sequence_buffer)
            pad_tensor = torch.zeros(1, pad_len, 3, device=self.device)
            pad_tensor[:, :, 2] = 0.0  # Set visibility flag to 0 for padding
            pad_tensor[:, :, 0] = -1.0  # Set position to -1 for padding
            pad_tensor[:, :, 1] = -1.0
            seq_input = torch.cat([pad_tensor, seq_input], dim=1)

        # Predict
        with torch.no_grad():
            prediction, _ = self.model(seq_input)
            pred = prediction.cpu().numpy()[0]  # (2,)

        return pred


def create_gru_predictor(device, seq_len=None, hidden_dim=64, num_layers=2):
    """
    Create a GRU predictor instance.

    Args:
        device: torch device
        seq_len: Sequence length (default from NetParameters.CONTEXT_LEN)
        hidden_dim: GRU hidden dimension
        num_layers: Number of GRU layers

    Returns:
        gru_model: GRUPredictor instance
        wrapper: GRUPredictorWrapper instance
    """
    if seq_len is None:
        seq_len = NetParameters.CONTEXT_LEN

    model = GRUPredictor(
        seq_len=seq_len,
        input_dim=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=2
    ).to(device)

    wrapper = GRUPredictorWrapper(model, device, seq_len=seq_len)

    return model, wrapper
