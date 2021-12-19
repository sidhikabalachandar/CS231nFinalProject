# adapted from code provided in Stanford CS231N and CS236

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import open3d as o3d

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class MaskedLinear(nn.Linear):

    def __init__(self, input_size, output_size, mask):
        super().__init__(input_size, output_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class PermuteLayer(nn.Module):

    def __init__(self, num_inputs):
        super(PermuteLayer, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])

    def forward(self, inputs, forward):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device
        )

    def inverse(self, inputs):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device
        )


class MADE(nn.Module):

    def __init__(self, input_size, hidden_size, n_hidden):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden

        masks = self.create_masks()

        self.net = [MaskedLinear(self.input_size, self.hidden_size, masks[0])]
        self.net += [nn.ReLU(inplace=True)]
        for i in range(self.n_hidden):
            self.net += [MaskedLinear(self.hidden_size, self.hidden_size, masks[i + 1])]
            self.net += [nn.ReLU(inplace=True)]
        self.net += [
            MaskedLinear(self.hidden_size, self.input_size * 2, masks[-1].repeat(2, 1))
        ]
        self.net = nn.Sequential(*self.net)

    def create_masks(self):
        masks = []
        input_degrees = torch.arange(self.input_size)
        degrees = [input_degrees]  # corresponds to m(k) in paper

        for n_h in range(self.n_hidden + 1):
            degrees += [torch.arange(self.hidden_size) % (self.input_size - 1)]
        degrees += [input_degrees % self.input_size - 1]
        self.m = degrees

        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

        return masks

    def forward(self, z, device):
        x = torch.zeros_like(z)

        b, n = z.size()
        log_det = torch.zeros((b)).to(device)
        for i in range(n):
            out = self.net(x)
            mu = out[:, :self.input_size]
            mu = mu[:, i]
            alpha = out[:, self.input_size:]
            alpha = alpha[:, i]
            log_det += alpha
            x[:, i] = mu + z[:, i] * torch.exp(alpha)

        log_det = -log_det

        return x, log_det

    def inverse(self, x):
        out = self.net(x)
        mu = out[:, :self.input_size]
        alpha = out[:, self.input_size:]
        z = (x - mu) / torch.exp(alpha)
        log_det = -torch.sum(alpha, dim=1)

        return z, log_det


class MAF(nn.Module):

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
        b, n = x.size()
        log_prob = torch.zeros((b)).to(device)
        z = x
        for module in self.nf:
            z, log_det = module.inverse(z)
            log_prob += log_det.squeeze()

        lpz = self.base_dist.log_prob(z)
        log_prob += lpz.sum(dim=1)
        log_prob = torch.mean(log_prob)
        return log_prob

    def loss(self, x, device):
        return -self.log_probs(x, device)

    def sample(self, device, n):
        with torch.no_grad():
            x_sample = torch.randn(n, self.input_size).to(device)
            for flow in self.nf[::-1]:
                x_sample, log_det = flow.forward(x_sample, device)
            x_sample = x_sample.view(n, self.input_size)

        return x_sample

def load_ae(ae_name):
    model = torch.load(ae_name)
    model.eval()

    return model

def encode(model, x):
    latent_code = x

    for layer in list(model.children())[:18]:
        latent_code = layer(latent_code)

    return latent_code

def decode(model, latent_code):
    x = latent_code

    for layer in list(model.children())[18:]:
        x = layer(x)

    return x

def run_a_maf(maf_model, maf_optimizer, loader_train, ae_name, device,
              batch_size=128, num_epochs=10, saved_models="saved_models",
              folder_name="folder_name", path_loss="path_loss", generated_samples_folder="Generated_Samples"):

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


                torch.save(maf_model, "{}/MAF_{}.pt".format(os.path.join(saved_models, folder_name), epoch))
    file_handle.close()

    return images
