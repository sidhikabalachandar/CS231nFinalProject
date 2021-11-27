import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class MaskedLinear(nn.Linear):
    """Masked linear layer for MADE: takes in mask as input and masks out connections in the linear layers."""

    def __init__(self, input_size, output_size, mask):
        super().__init__(input_size, output_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class PermuteLayer(nn.Module):
    """Layer to permute the ordering of inputs.

    Because our data is 2-D, forward() and inverse() will reorder the data in the same way.
    """

    def __init__(self, num_inputs):
        super(PermuteLayer, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])

    def forward(self, inputs):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device
        )

    def inverse(self, inputs):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device
        )


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation.
    https://arxiv.org/abs/1502.03509

    Uses sequential ordering as in the MAF paper.
    Gaussian MADE to work with real-valued inputs"""

    def __init__(self, input_size, hidden_size, n_hidden):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden

        masks = self.create_masks()

        # construct layers: inner, hidden(s), output
        self.net = [MaskedLinear(self.input_size, self.hidden_size, masks[0])]
        self.net += [nn.ReLU(inplace=True)]
        # iterate over number of hidden layers
        for i in range(self.n_hidden):
            self.net += [MaskedLinear(self.hidden_size, self.hidden_size, masks[i + 1])]
            self.net += [nn.ReLU(inplace=True)]
        # last layer doesn't have nonlinear activation
        self.net += [
            MaskedLinear(self.hidden_size, self.input_size * 2, masks[-1].repeat(2, 1))
        ]
        self.net = nn.Sequential(*self.net)

    def create_masks(self):
        """
        Creates masks for sequential (natural) ordering.
        """
        masks = []
        input_degrees = torch.arange(self.input_size)
        degrees = [input_degrees]  # corresponds to m(k) in paper

        # iterate through every hidden layer
        for n_h in range(self.n_hidden + 1):
            degrees += [torch.arange(self.hidden_size) % (self.input_size - 1)]
        degrees += [input_degrees % self.input_size - 1]
        self.m = degrees

        # output layer mask
        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

        return masks

    def forward(self, z):
        """
        Run the forward mapping (z -> x) for MAF through one MADE block.
        :param z: Input noise of size (batch_size, self.input_size)
        :return: (x, log_det). log_det should be 1-D (batch_dim,)
        """
        x = torch.zeros_like(z)

        # YOUR CODE STARTS HERE
        b, n = z.size()
        log_det = torch.zeros((b))
        for i in range(n):
            out = self.net(x)
            mu = out[:, :2]
            mu = mu[:, i]
            alpha = out[:, 2:]
            alpha = alpha[:, i]
            log_det += alpha
            x[:, i] = mu + z[:, i] * torch.exp(alpha)

        log_det = -log_det
        # YOUR CODE ENDS HERE

        return x, log_det

    def inverse(self, x):
        """
        Run one inverse mapping (x -> z) for MAF through one MADE block.
        :param x: Input data of size (batch_size, self.input_size)
        :return: (z, log_det). log_det should be 1-D (batch_dim,)
        """
        # YOUR CODE STARTS HERE
        out = self.net(x)
        mu = out[:, :self.input_size]
        alpha = out[:, self.input_size:]
        z = (x - mu) / torch.exp(alpha)
        log_det = -torch.sum(alpha, dim=1)
        # YOUR CODE ENDS HERE

        return z, log_det


class MAF(nn.Module):
    """
    Masked Autoregressive Flow, using MADE layers.
    https://arxiv.org/abs/1705.07057
    """

    def __init__(self, input_size, hidden_size, n_hidden, n_flows):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.n_flows = n_flows
        self.base_dist = torch.distributions.normal.Normal(0, 1)

        # need to flip ordering of inputs for every layer
        nf_blocks = []
        for i in range(self.n_flows):
            nf_blocks.append(MADE(self.input_size, self.hidden_size, self.n_hidden))
            nf_blocks.append(PermuteLayer(self.input_size))  # permute dims
        self.nf = nn.Sequential(*nf_blocks)

    def log_probs(self, x, device):
        """
        Obtain log-likelihood p(x) through one pass of MADE
        :param x: Input data of size (batch_size, self.input_size)
        :return: log_prob. This should be a Python scalar.
        """
        # YOUR CODE STARTS HERE
        b, n = x.size()
        log_prob = torch.zeros((b)).to(device)
        z = x
        for module in self.nf:
            z, log_det = module.inverse(z)
            log_prob += log_det.squeeze()

        lpz = self.base_dist.log_prob(z)
        log_prob += lpz.sum(dim=1)
        log_prob = torch.mean(log_prob)
        # YOUR CODE ENDS HERE

        return log_prob

    def loss(self, x, device):
        """
        Compute the loss.
        :param x: Input data of size (batch_size, self.input_size)
        :return: loss. This should be a Python scalar.
        """
        return -self.log_probs(x, device)

    def sample(self, device, n):
        """
        Draw <n> number of samples from the model.
        :param device: [cpu,cuda]
        :param n: Number of samples to be drawn.
        :return: x_sample. This should be a numpy array of size (n, self.input_size)
        """
        with torch.no_grad():
            x_sample = torch.randn(n, self.input_size).to(device)
            for flow in self.nf[::-1]:
                x_sample, log_det = flow.forward(x_sample)
            x_sample = x_sample.view(n, self.input_size)
            #x_sample = x_sample.cpu().data.numpy()

        return x_sample

def load_ae(ae_name):
    model = torch.load(ae_name)
    model.eval()

    return model

def encode(model, x):

    # model - this the auto encoder (we only use the encoding half)
    # x (batch_size, 2048, 3)

    latent_code = x

    # Extract Latent 128 vector ONLY up to 18 layers
    for layer in list(model.children())[:18]:
        latent_code = layer(latent_code)

    return latent_code

def decode(model, latent_code):

    # model - this the auto encoder (we only use the decoding half)
    # latent_code (batch_size, 128)

    x = latent_code

    # Extract Full Point Cloud starting from Latent 128 vector
    for layer in list(model.children())[18:]:
        x = layer(x)

    return x

def run_a_maf(maf_model, maf_optimizer, loader_train, ae_name, device,
              show_every=250, batch_size=128, num_epochs=10, saved_models="saved_models",
              folder_name="folder_name", path_loss="path_loss", generated_samples_folder="Generated_Samples"):
    """
    Train a GAN!

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """

    file_handle = open(path_loss, "a")

    os.makedirs(os.path.join(saved_models, folder_name), exist_ok=True)
    os.makedirs(os.path.join(generated_samples_folder, folder_name), exist_ok=True)

    images = []

    ae = load_ae(ae_name)

    maf_model.train()
    total_loss = 0.0
    batch_idx = 0.0

    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue

            batch_idx += 1
            real_data = x.type(dtype)
            real_data = torch.reshape(real_data, (-1, 2048, 3))
            real_data = real_data.transpose(1, 2)
            real_data = encode(ae, real_data)
            loss = maf_model.loss(real_data, device)

            maf_optimizer.zero_grad()
            loss.backward()
            maf_optimizer.step()

            # save stuff
            total_loss += loss.item()

        total_loss /= batch_idx + 1
        print("Average train log-likelihood: {:.6f}".format(-total_loss))

        # display the epoch training and validatation loss
        epoch_str = "epoch : {}/{}, Average train log-likelihood: {:.6f}".format(epoch + 1, num_epochs, -total_loss)
        file_handle.write(epoch_str + "\n")

        if (epoch + 1) % 50 == 0:
            fake_images = maf_model.sample(device, n=4)
            output = decode(ae, fake_images.data)
            imgs_numpy = output.cpu().detach().numpy()
            imgs_numpy = imgs_numpy.reshape(-1, 2048, 3)

            for sample in range(0, 4):
                curr_pc = imgs_numpy[sample, :, :]
                pcd_output = o3d.geometry.PointCloud()
                pcd_output.points = o3d.utility.Vector3dVector(curr_pc)
                o3d.io.write_point_cloud(os.path.join(generated_samples_folder, folder_name,
                                                      "epoch_{}_sample_{}.ply".format(epoch + 1, sample)),
                                         pcd_output)

            torch.save(G, "{}/generator_{}.pt".format(os.path.join(saved_models, folder_name), epoch))
            torch.save(D, "{}/discriminator_{}.pt".format(os.path.join(saved_models, folder_name), epoch))

    file_handle.close()

    return images