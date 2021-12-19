# adapted from code provided in Stanford CS231N and CS236

import os
import torch
import torch.nn as nn
import open3d as o3d

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def sample_noise(batch_size, dim):
    return torch.normal(0, 0.2, (batch_size, dim))


def rgan_discriminator(input_dim=3):
    # input  : (N, 3, 2048)
    # output : (N, 1)

    model = nn.Sequential(
        nn.Conv1d(input_dim, 64, 1),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.2),
        nn.Conv1d(64, 128, 1),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(0.2),
        nn.Conv1d(128, 512, 1),
        nn.BatchNorm1d(512),
        nn.MaxPool1d(2048),
        nn.Flatten(),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.2),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    return model

def lgan_discriminator(input_dim=128):
    # input  : (N, 3, 2048)
    # output : (N, 1)

    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=256, out_features=512),
        nn.BatchNorm1d(num_features=512),
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=512, out_features=1),
        nn.Sigmoid()
    )
    return model


def rgan_generator(noise_dim=128):
    model = nn.Sequential(
        nn.Linear(noise_dim, 64),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=64, out_features=128),
        nn.BatchNorm1d(num_features=128),
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=128, out_features=512),
        nn.BatchNorm1d(num_features=512),
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=512, out_features=6144)
    )
    return model


def lgan_generator(noise_dim=128):
    model = nn.Sequential(
        nn.Linear(noise_dim, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(0.2),
        nn.Linear(in_features=128, out_features=128)
    )
    return model


def bce_loss(input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def discriminator_loss(logits_real, logits_fake):
    N = logits_real.size()
    true_labels = torch.ones(N).type(dtype)
    false_labels = torch.zeros(N).type(dtype)
    loss_real = bce_loss(logits_real, true_labels)
    loss_fake = bce_loss(logits_fake, false_labels)
    loss = loss_real + loss_fake

    return loss


def generator_loss(logits_fake):
    N = logits_fake.size()
    true_labels = torch.ones(N).type(dtype)
    loss = bce_loss(logits_fake, true_labels)

    return loss


def loss_wasserstein_gp_d(g, d, x_real, noise_size, device):
    batch_size = x_real.shape[0]
    z = sample_noise(batch_size, noise_size).type(dtype)

    term_1 = torch.mean(d(g(z)))
    term_2 = torch.mean(d(x_real))
    alpha = torch.rand((batch_size)).to(device)
    r = alpha * g(z).permute((1, 0)) + (1 - alpha) * x_real.permute((1, 0))
    r = r.permute(1, 0)
    lda = 10
    grad = torch.autograd.grad(d(r).sum(), r, create_graph=True)
    grad = torch.linalg.norm(grad[0], dim=1)
    term_3 = lda * torch.mean((grad - 1) ** 2)
    d_loss = term_1 - term_2 + term_3

    return d_loss


def loss_wasserstein_gp_g(g, d, x_real, noise_size):
    batch_size = x_real.shape[0]
    z = sample_noise(batch_size, noise_size).type(dtype)

    g_loss = -torch.mean(d(g(z)))

    return g_loss


def load_ae(ae_name):
    model = torch.load(ae_name)
    model.eval()

    return model


def encode(model, x):
    
    # model - this the auto encoder (we only use the encoding half)
    # x (batch_size, 2048, 3)

    latent_code = x

    #Extract Latent 128 vector ONLY up to 18 layers
    for layer in list(model.children())[:18]:
        latent_code = layer(latent_code)

    return latent_code

def decode(model, latent_code):
    
    # model - this the auto encoder (we only use the decoding half)
    # latent_code (batch_size, 128)

    x = latent_code

    #Extract Full Point Cloud starting from Latent 128 vector
    for layer in list(model.children())[18:]:
        x = layer(x)

    return x


def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader_train, do_lgan, ae_name,
                show_every=250, batch_size=128, noise_size=96, num_epochs=10, saved_models="saved_models", 
                folder_name="folder_name", path_loss="path_loss", generated_samples_folder="Generated_Samples"):
    file_handle = open(path_loss, "a")

    os.makedirs(os.path.join(saved_models, folder_name), exist_ok=True)
    os.makedirs(os.path.join(generated_samples_folder, folder_name), exist_ok=True)

    images = []
    iter_count = 0


    if do_lgan:
        ae = load_ae(ae_name)

    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.type(dtype)
            real_data = torch.reshape(real_data, (-1, 2048, 3))
            real_data = real_data.transpose(1, 2)
            if do_lgan:
                real_data = encode(ae, real_data)
            logits_real = D(real_data).type(dtype)

            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed).detach()
            if do_lgan:
                logits_fake = D(fake_images)
            else:
                logits_fake = D(fake_images.view(batch_size, 2048, 3).transpose(1, 2))

            d_error = discriminator_loss(logits_real, logits_fake)
            d_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)

            if do_lgan:
                gen_logits_fake = D(fake_images)
            else:
                gen_logits_fake = D(fake_images.view(batch_size, 2048, 3).transpose(1, 2))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if iter_count % show_every == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_error.item(), g_error.item()))

            iter_count += 1


        # display the epoch training and validatation loss
        epoch_str = "epoch : {}/{}, discriminator_loss = {:.4f}, generator_loss = {:.4f}".format(epoch + 1, num_epochs, d_error.item(), g_error.item())
        file_handle.write(epoch_str + "\n")

        if (epoch + 1) % 50 == 0:
            output = fake_images.data
            if do_lgan:
                output = decode(ae, output)
            imgs_numpy = output.cpu().detach().numpy()
            imgs_numpy = imgs_numpy.reshape(-1, 2048, 3)

            for sample in range(0,4):
                curr_pc = imgs_numpy[sample,:,:]
                pcd_output = o3d.geometry.PointCloud()
                pcd_output.points = o3d.utility.Vector3dVector(curr_pc)
                o3d.io.write_point_cloud(os.path.join(generated_samples_folder, folder_name, "epoch_{}_sample_{}.ply".format(epoch+1, sample)), pcd_output)

            torch.save(G, "{}/generator_{}.pt".format(os.path.join(saved_models, folder_name), epoch))
            torch.save(D, "{}/discriminator_{}.pt".format(os.path.join(saved_models, folder_name), epoch))

    file_handle.close()

    return images


def run_a_wgan(D, G, D_solver, G_solver, d_loss, g_loss, loader_train, ae_name, device,
                show_every=250, batch_size=128, noise_size=96, num_epochs=10, saved_models="saved_models",
                folder_name="folder_name", path_loss="path_loss", generated_samples_folder="Generated_Samples"):
    file_handle = open(path_loss, "a")

    os.makedirs(os.path.join(saved_models, folder_name), exist_ok=True)
    os.makedirs(os.path.join(generated_samples_folder, folder_name), exist_ok=True)

    images = []
    iter_count = 0


    ae = load_ae(ae_name)

    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.type(dtype)
            real_data = torch.reshape(real_data, (-1, 2048, 3))
            real_data = real_data.transpose(1, 2)
            real_data = encode(ae, real_data) # batch size x 128
            d_error = d_loss(G, D, real_data, noise_size, device)
            d_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            g_error = g_loss(G, D, real_data, noise_size)
            g_error.backward()
            G_solver.step()

            if iter_count % show_every == 0:
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_error.item(), g_error.item()))

            iter_count += 1


        # display the epoch training and validatation loss
        epoch_str = "epoch : {}/{}, discriminator_loss = {:.4f}, generator_loss = {:.4f}".format(epoch + 1, num_epochs, d_error.item(), g_error.item())
        file_handle.write(epoch_str + "\n")

        if (epoch + 1) % 50 == 0:
            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed)
            output = fake_images.data
            output = decode(ae, output)
            imgs_numpy = output.cpu().detach().numpy()
            imgs_numpy = imgs_numpy.reshape(-1, 2048, 3)

            for sample in range(0,4):
                curr_pc = imgs_numpy[sample,:,:]
                pcd_output = o3d.geometry.PointCloud()
                pcd_output.points = o3d.utility.Vector3dVector(curr_pc)
                o3d.io.write_point_cloud(os.path.join(generated_samples_folder, folder_name, "epoch_{}_sample_{}.ply".format(epoch+1, sample)), pcd_output)

            torch.save(G, "{}/generator_{}.pt".format(os.path.join(saved_models, folder_name), epoch))
            torch.save(D, "{}/discriminator_{}.pt".format(os.path.join(saved_models, folder_name), epoch))

    file_handle.close()

    return images

