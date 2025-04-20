
from .net import *
from .encoder import *
from .decoder import *


class VariationalAutoencoder(Net):
    """
    Base class to build the variational autoencoder and manage output from encoder and decoder.
    """
    def __init__(self, latent_dims: int, distance: int, name: str, structure: str, noise: str, cluster: bool = False, device: torch.device = torch.device('cpu')):
        super(VariationalAutoencoder, self).__init__(name, cluster)

        # Number of input channels
        if noise == 'bitflip':
            channels = 1
        elif noise == 'ising':
            channels = 1
        elif noise == 'depolarizing':
            channels = 2
        else:
            raise ValueError('Invalid noise for this autoencoder structure.')
        assert channels > 0

        self.structure = structure
        # Specify structure of the VAE
        if structure == 'upsampling':
            self.encoder = VariationalEncoder(latent_dims, distance, channels, device=device)
            self.decoder = Decoder(latent_dims, distance, channels, device=device)
        elif structure == 'conv-only':
            self.encoder = VariationalEncoderIsing(latent_dims, distance, channels)
            self.decoder = DecoderIsing(latent_dims, distance, channels)
        elif structure == 'skip':
            self.encoder = VariationalEncoderSkip(latent_dims, distance, channels, device=device)
            self.decoder = DecoderSkip(latent_dims, distance, channels, device=device)
        '''
        if structure == 'standard':
            self.encoder = VariationalEncoder(latent_dims, distance, channels, device=device)
            self.decoder = Decoder(latent_dims, distance, channels, device=device)
        elif structure == 'simple':
            self.encoder = VariationalEncoderSimple(latent_dims, distance, channels)
            self.decoder = DecoderSimple(latent_dims, distance, channels)
        elif structure == 'skip':
            self.encoder = VariationalEncoder(latent_dims, distance, channels)
            self.decoder = DecoderSkip(latent_dims, distance, channels)
        elif structure == 'deep-skip':
            self.encoder = VariationalEncoderDeep(latent_dims, distance)
            self.decoder = DecoderDeepSkip(latent_dims, distance)
        elif structure == 'deep':
            self.encoder = VariationalEncoderDeep(latent_dims, distance)
            self.decoder = DecoderDeep(latent_dims, distance)
        elif structure == 'ising':
            self.encoder = VariationalEncoderIsing(latent_dims, distance, channels)
            self.decoder = DecoderIsing(latent_dims, distance, channels)
        elif structure == 'transformer':
            self.encoder = TransformerEncoder(latent_dims, distance, channels)
            self.decoder = TransformerDecoder(latent_dims, distance, channels)
        else:
            raise Exception('Invalid structure specified.')
        '''

    def forward(self, x):
        z_mean, z_log_var, z, skip = self.encoder(x)
        x = self.decoder(z, skip)
        return x, z_mean, z_log_var


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, hidden_dim, dev, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(input_size=d_model, hidden_size=hidden_dim, batch_first=True, bidirectional=False, device=dev)

    def forward(self, x):
        x = x + self.dropout(self.rnn(x)[0])
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n, d_model):
        # put n here as max_seq_len
        super().__init__()
        self.positional_embedding = nn.Embedding(n, d_model)

    def forward(self, x):
        return x + self.positional_embedding(torch.arange(x.size(1), device=x.device))


class TraVAE(Net):
    """
    Transformer-based variational autoencoder to handle sequential data.
    """
    def __init__(self, latent_dims: int, distance: int, name: str, *args, cluster: bool = False, **kwargs):
        super(TraVAE, self).__init__(name, cluster)
        self.n = kwargs['n']  # number of data qubits
        self.k = kwargs['k']  # number of logical qubits -> number of syndromes: n - k
        self.d_model = kwargs['d_model']  # dimension of embedding
        self.d_ff = kwargs['d_ff']  # dimension of feedforward layer
        self.n_layers = kwargs['n_layers']  # number of attention layers
        self.n_heads = kwargs['n_heads']  # number of attention heads
        self.dropout = kwargs['dropout']  # dropout rate
        self.device = kwargs['device']  # device
        self.seq_len = self.n - 1 + 2 * distance * self.k  + 1
        self.latent_dims = latent_dims

        self.fc_in = nn.Embedding(2, self.d_model)
        # self.positional_encoding = PositionalEncoding(self.d_model, self.d_model, self.device, self.dropout)
        self.positional_encoding = LearnablePositionalEncoding(self.n - 1 + 6 * self.k, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=self.dropout,
                                                   batch_first=True,
                                                   device=self.device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.fc_enc = nn.Linear(self.d_model, 1)
        # Mean and log-variance layers for latent space
        self.mean_fc = nn.Linear(self.seq_len, latent_dims)
        self.logvar_fc = nn.Linear(self.seq_len, latent_dims)

        # Transformer decoder (autoregressive) TODO think about how decoder could and would look like
        decoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=self.dropout,
                                                   batch_first=True,
                                                   device=self.device)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=self.n_layers)

        '''Here attempt, where latent variable as first tokens -> cf. transformer vae password generation paper'''

        # Latent to embedding layer
        self.fc_dec = nn.Linear(latent_dims, self.seq_len)
        self.latent_to_embedding = nn.Linear(1, self.d_model)
        self.positional_encoding_dec = LearnablePositionalEncoding(self.n - 1 + 6 * self.k + self.latent_dims, self.d_model)

        '''Here attempt as in sentence generation paper, output completely generated by latent variable vector'''
        # self.linear_proj = nn.Linear(self.latent_dims, self.d_model)
        # self.rnn = nn.GRU(input_size=self.d_model, hidden_size=1, batch_first=True, bidirectional=False, device=self.device)

        # Output layer to map back to vocabulary
        self.fc_out = nn.Linear(self.d_model, 1)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, src):
        """Autoregressive encoding"""
        # seq_len = src.size(1)
        # mask = torch.ones(seq_len, seq_len)
        # mask = torch.tril(mask, diagonal=0)

        # For subsequent tokens, set diagonal values to 0 to prevent self-attention (except for the start token)
        # mask[1:, 1:] = torch.tril(torch.ones((seq_len - 1, seq_len - 1)), diagonal=-1)

        # mask = mask.masked_fill(mask == 0, float('-inf'))
        # mask = mask.masked_fill(mask == 1, 0.0)
        # mask = mask.to(self.device)

        memory = self.encoder(src)  #, mask=mask)
        encoded = self.fc_enc(memory).squeeze(-1)

        # Use final output for mean and variance
        mean = self.mean_fc(encoded)  # mean across tokens
        logvar = self.logvar_fc(encoded)  # log variance across tokens
        return mean, logvar  #, memory

    def decode(self, x, z):  #, memory):
        """Autoregressive decoding"""
        # Transform the latent vector to start token embeddings
        start_token_value = 1
        start_token = torch.full((x.size(0), 1), start_token_value, dtype=torch.long, device=self.device)
        tgt = torch.cat((z, start_token, x[:, :-1]), dim=1).unsqueeze(-1)
        # z = self.fc_dec(z).unsqueeze(-1)
        tgt = self.latent_to_embedding(tgt)
        # tgt = z.unsqueeze(1).expand(-1, x.size(1), -1)
        # tgt = self.linear_proj(tgt)
        tgt = self.positional_encoding_dec(tgt)

        seq_len = tgt.size(1)
        mask = torch.ones(seq_len, seq_len)
        mask = torch.tril(mask, diagonal=0)

        mask[:self.latent_dims, :self.latent_dims] = torch.full((self.latent_dims, self.latent_dims), 1)
        # For subsequent tokens, set diagonal values to 0 to prevent self-attention (except for the start token)
        mask[self.latent_dims + 1:, self.latent_dims + 1:] = torch.tril(torch.ones((seq_len - self.latent_dims - 1, seq_len - self.latent_dims - 1)), diagonal=-1)

        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        mask = mask.to(self.device)

        output = self.decoder(tgt, mask=mask)  # , memory_mask=mask)
        # output = self.rnn(output)[0]
        return F.sigmoid(self.fc_out(output))
        # return F.sigmoid(output)

    def forward(self, input):
        # Encoding
        src = input

        start_token_value = 1
        start_token = torch.full((src.size(0), 1), start_token_value, dtype=torch.long, device=self.device)
        src = torch.cat((start_token, src), dim=1).to(self.device)  # here not src[:, :-1]

        src = F.relu(self.fc_in(src))
        src = self.positional_encoding(src)
        mean, logvar = self.encode(src)
        z = self.reparameterize(mean, logvar)

        # Decoding
        output = self.decode(input, z)
        return output[:, self.latent_dims:, :], mean, logvar, z
        # return output, mean, logvar, z

    def partial_forward(self, src):
        # Encoding
        start_token_value = 1
        start_token = torch.full((src.size(0), 1), start_token_value, dtype=torch.long, device=self.device)
        src = torch.cat((start_token, src), dim=1).to(self.device)  # here not src[:, :-1]

        src = F.relu(self.fc_in(src))
        src = self.positional_encoding(src)
        mean, logvar = self.encode(src)
        return mean




