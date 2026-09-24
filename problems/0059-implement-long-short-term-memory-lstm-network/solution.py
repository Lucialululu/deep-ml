import torch

class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases as float64 tensors
        self.Wf = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wi = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wc = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wo = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)

        self.bf = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bi = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bc = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bo = torch.zeros(hidden_size, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor, initial_hidden_state: torch.Tensor, initial_cell_state: torch.Tensor):
        """
        Processes a sequence of inputs and returns the hidden states,
        final hidden state, and final cell state.

        Args:
            x: Input tensor of shape (seq_len, input_size)
            initial_hidden_state: Initial hidden state of shape (hidden_size, 1)
            initial_cell_state: Initial cell state of shape (hidden_size, 1)

        Returns:
            outputs: Tensor of hidden states at each time step
            h: Final hidden state tensor
            c: Final cell state tensor
        """
        h = initial_hidden_state
        c = initial_cell_state

        outputs = []

        seq_length = x.shape[0]

        for t in range(seq_length):
            x_t = x[t].reshape(self.input_size, 1)
            combined_vector = torch.cat([h, x_t], dim=0)

            f_t = torch.sigmoid(self.Wf @ combined_vector + self.bf)
            i_t = torch.sigmoid(self.Wi @ combined_vector + self.bi)

            tilde_c_t = torch.tanh(self.Wc @ combined_vector + self.bc)

            o_t = torch.sigmoid(self.Wo @ combined_vector + self.bo)

            c = f_t * c + i_t * tilde_c_t

            h = o_t * torch.tanh(c) 

            outputs.append(h)

        outputs = torch.stack(outputs, dim=0)

        return outputs, h, c

