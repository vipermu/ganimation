import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from model import Generator
from model import Discriminator
from utils import InitializerClass, UtilsClass

import numpy as np

import os
import time
import datetime
import random
import glob


class Solver(InitializerClass, UtilsClass):
    """Solver for training and testing StarGAN."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""
        super().__init__(config)
        self.data_loader = data_loader

    def train(self):
        print('Training the model...')

        # Start training from scratch or resume training.
        if self.resume_iters:
            self.first_iteration = self.resume_iters
            self.restore_model(self.resume_iters)

        self.start_time = time.time()

        for epoch in range(self.first_epoch, self.num_epochs):
            print(f"EPOCH {epoch} WITH {len(self.data_loader)} STEPS")
            self.alpha_rec = 1
            self.epoch = epoch

            for iteration in range(self.first_iteration, len(self.data_loader)):
                self.iteration = iteration
                self.get_training_data()
                self.train_discriminator()

                if (self.iteration+1) % self.n_critic == 0:
                    generation_outputs = self.train_generator()

                if (self.iteration+1) % self.sample_step == 0:
                    self.print_generations(generation_outputs)
                    # self.save_generation_matrices()

                if self.iteration % self.model_save_step == 0:
                    self.save_models(self.iteration, self.epoch)

                if self.iteration % self.log_step == 0:
                    self.update_tensorboard()
                self.global_counter += 1

            # Decay learning rates.
            if (self.epoch+1) > self.num_epochs_decay:
                # float(self.num_epochs_decay))
                self.g_lr -= (self.g_lr / 10.0)
                # float(self.num_epochs_decay))
                self.d_lr -= (self.d_lr / 10.0)
                self.update_lr(self.g_lr, self.d_lr)
                print('Decayed learning rates, self.g_lr: {}, self.d_lr: {}.'.format(
                    self.g_lr, self.d_lr))

            # Save the last model
            self.save_models()

            self.first_iteration = 0  # Next epochs start from 0

    def get_training_data(self):
        try:
            self.x_real, self.label_org = next(self.data_iter)
        except:
            self.data_iter = iter(self.data_loader)
            self.x_real, self.label_org = next(self.data_iter)

        self.x_real = self.x_real.to(self.device)  # Input images.
        # Labels for computing classification loss.
        self.label_org = self.label_org.to(self.device)

        # Get random targets for training
        self.label_trg = self.get_random_labels_list()
        self.label_trg = torch.FloatTensor(self.label_trg).clamp(0, 1)
        # Labels for computing classification loss.
        self.label_trg = self.label_trg.to(self.device)

        if self.use_virtual:
            self.label_trg_virtual = self.get_random_labels_list()
            self.label_trg_virtual = torch.FloatTensor(
                self.label_trg_virtual).clamp(0, 1)
            # Labels for computing classification loss.
            self.label_trg_virtual = self.label_trg_virtual.to(self.device)

            assert not torch.equal(
                self.label_trg_virtual, self.label_trg), "Target label and virtual label are the same"

    def get_random_labels_list(self):
        trg_list = []
        for _ in range(self.batch_size):
            random_num = random.randint(
                0, len(self.data_loader)*self.batch_size-1)
            # Select a random AU vector from the dataset
            trg_list_aux = self.data_loader.dataset[random_num][1]
            # Apply a variance of 0.1 to the vector
            trg_list.append(trg_list_aux.numpy() +
                            np.random.uniform(-0.1, 0.1, trg_list_aux.shape))
        return trg_list

    def train_discriminator(self):
        # Compute loss with real images.
        critic_output, classification_output = self.D(self.x_real)
        d_loss_critic_real = -torch.mean(critic_output)
        d_loss_classification = torch.nn.functional.mse_loss(
            classification_output, self.label_org)

        # Compute loss with fake images.
        attention_mask, color_regression = self.G(self.x_real, self.label_trg)
        x_fake = self.imFromAttReg(
            attention_mask, color_regression, self.x_real)
        critic_output, _ = self.D(x_fake.detach())
        d_loss_critic_fake = torch.mean(critic_output)

        # Compute loss for gradient penalty.
        alpha = torch.rand(self.x_real.size(0), 1, 1, 1).to(self.device)
        # Half of image info from fake and half from real
        x_hat = (alpha * self.x_real.data + (1 - alpha)
                 * x_fake.data).requires_grad_(True)
        critic_output, _ = self.D(x_hat)
        d_loss_gp = self.gradient_penalty(critic_output, x_hat)

        # Backward and optimize.
        d_loss = d_loss_critic_real + d_loss_critic_fake + self.lambda_cls * \
            d_loss_classification + self.lambda_gp * d_loss_gp

        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Logging.
        self.loss_visualization['D/loss'] = d_loss.item()
        self.loss_visualization['D/loss_real'] = d_loss_critic_real.item()
        self.loss_visualization['D/loss_fake'] = d_loss_critic_fake.item()
        self.loss_visualization['D/loss_cls'] = self.lambda_cls * \
            d_loss_classification.item()
        self.loss_visualization['D/loss_gp'] = self.lambda_gp * \
            d_loss_gp.item()

    def train_generator(self):
        # Original-to-target domain.
        attention_mask, color_regression = self.G(self.x_real, self.label_trg)
        x_fake = self.imFromAttReg(
            attention_mask, color_regression, self.x_real)

        critic_output, classification_output = self.D(x_fake)
        g_loss_fake = -torch.mean(critic_output)
        g_loss_cls = torch.nn.functional.mse_loss(
            classification_output, self.label_trg)

        # Target-to-original domain.
        if not self.use_virtual:
            reconstructed_attention_mask, reconstructed_color_regression = self.G(
                x_fake, self.label_org)
            x_rec = self.imFromAttReg(
                reconstructed_attention_mask, reconstructed_color_regression, x_fake)

        else:
            reconstructed_attention_mask, reconstructed_color_regression = self.G(
                x_fake, self.label_org)
            x_rec = self.imFromAttReg(
                reconstructed_attention_mask, reconstructed_color_regression, x_fake)

            reconstructed_attention_mask_2, reconstructed_color_regression_2 = self.G(
                x_fake, self.label_trg_virtual)
            x_fake_virtual = self.imFromAttReg(
                reconstructed_attention_mask_2, reconstructed_color_regression_2, x_fake)

            reconstructed_virtual_attention_mask, reconstructed_virtual_color_regression = self.G(
                x_fake_virtual, self.label_trg)
            x_rec_virtual = self.imFromAttReg(
                reconstructed_virtual_attention_mask, reconstructed_virtual_color_regression, x_fake_virtual.detach())

        # Compute losses
        g_loss_saturation_1 = attention_mask.mean()
        g_loss_smooth1 = self.smooth_loss(attention_mask)

        if not self.use_virtual:
            g_loss_rec = torch.nn.functional.l1_loss(self.x_real, x_rec)
            g_loss_saturation_2 = reconstructed_attention_mask.mean()
            g_loss_smooth2 = self.smooth_loss(reconstructed_attention_mask)

        else:
            g_loss_rec = (1-self.alpha_rec)*torch.nn.functional.l1_loss(self.x_real, x_rec) + \
                self.alpha_rec * \
                torch.nn.functional.l1_loss(x_fake, x_rec_virtual)

            g_loss_saturation_2 = (1-self.alpha_rec) * reconstructed_attention_mask.mean() + \
                self.alpha_rec * reconstructed_virtual_attention_mask.mean()

            g_loss_smooth2 = (1-self.alpha_rec) * self.smooth_loss(reconstructed_virtual_attention_mask) + \
                self.alpha_rec * self.smooth_loss(reconstructed_attention_mask)

        g_attention_loss = self.lambda_smooth * g_loss_smooth1 + self.lambda_smooth * g_loss_smooth2 \
            + self.lambda_sat * g_loss_saturation_1 + self.lambda_sat * g_loss_saturation_2

        g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + \
            self.lambda_cls * g_loss_cls + g_attention_loss

        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()

        # Logging.
        self.loss_visualization['G/loss'] = g_loss.item()
        self.loss_visualization['G/loss_fake'] = g_loss_fake.item()
        self.loss_visualization['G/loss_rec'] = self.lambda_rec * \
            g_loss_rec.item()
        self.loss_visualization['G/loss_cls'] = self.lambda_cls * \
            g_loss_cls.item()
        self.loss_visualization['G/attention_loss'] = g_attention_loss.item()
        self.loss_visualization['G/loss_smooth1'] = self.lambda_smooth * \
            g_loss_smooth1.item()
        self.loss_visualization['G/loss_smooth2'] = self.lambda_smooth * \
            g_loss_smooth2.item()
        self.loss_visualization['G/loss_sat1'] = self.lambda_sat * \
            g_loss_saturation_1.item()
        self.loss_visualization['G/loss_sat2'] = self.lambda_sat * \
            g_loss_saturation_2.item()
        self.loss_visualization['G/alpha'] = self.alpha_rec

        if not self.use_virtual:
            return {
                "color_regression": color_regression,
                "x_fake": x_fake,
                "attention_mask": attention_mask,
                "x_rec": x_rec,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_color_regression": reconstructed_color_regression,
            }

        else:
            return {
                "color_regression": color_regression,
                "x_fake": x_fake,
                "attention_mask": attention_mask,
                "x_rec": x_rec,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_color_regression": reconstructed_color_regression,
                "reconstructed_virtual_attention_mask": reconstructed_virtual_attention_mask,
                "reconstructed_virtual_color_regression": reconstructed_virtual_color_regression,
                "x_rec_virtual": x_rec_virtual,
            }

    def print_generations(self, generator_outputs_dict):
        print_epoch_images = False
        save_image(self.denorm(self.x_real), self.sample_dir +
                   '/{}_4real_.png'.format(self.epoch))
        save_image((generator_outputs_dict["color_regression"]+1)/2,
                   self.sample_dir + '/{}_2reg_.png'.format(self.epoch))
        save_image(self.denorm(
            generator_outputs_dict["x_fake"]), self.sample_dir + '/{}_3res_.png'.format(self.epoch))
        save_image(generator_outputs_dict["attention_mask"],
                   self.sample_dir + '/{}_1attention_.png'.format(self.epoch))
        save_image(self.denorm(
            generator_outputs_dict["x_rec"]), self.sample_dir + '/{}_5rec_.png'.format(self.epoch))
        if not self.use_virtual:
            save_image(generator_outputs_dict["reconstructed_attention_mask"],
                       self.sample_dir + '/{}_6rec_attention.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["reconstructed_color_regression"]), self.sample_dir + '/{}_7rec_reg.png'.format(self.epoch))
        else:
            save_image(generator_outputs_dict["reconstructed_attention_mask"],
                       self.sample_dir + '/{}_6rec_attention_.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["reconstructed_color_regression"]), self.sample_dir + '/{}_7rec_reg.png'.format(self.epoch))

            save_image(generator_outputs_dict["reconstructed_virtual_attention_mask"],
                       self.sample_dir + '/{}_8rec_virtual_attention.png'.format(self.epoch))
            save_image(self.denorm(generator_outputs_dict["reconstructed_virtual_color_regression"]),
                       self.sample_dir + '/{}_91rec_virtual_reg.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["x_rec_virtual"]), self.sample_dir + '/{}_92rec_epoch_.png'.format(self.epoch))

    def update_tensorboard(self):
        # Print out training information.
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}],  [{}/{}], Epoch [{}/{}]".format(
            et, self.iteration+1, len(self.data_loader), self.epoch+1, self.num_epochs)
        for tag, value in self.loss_visualization.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

        if self.use_tensorboard:
            for tag, value in self.loss_visualization.items():
                self.writer.add_scalar(
                    tag, value, global_step=self.global_counter)

    def save_generation_matrices(self):
        # Translate fixed images for debugging.
        if (self.iteration+1) % self.sample_step == 0:
            with torch.no_grad():
                x_real, real_labels = next(self.data_iter)
                x_real, real_labels = x_real.to(
                    self.device), real_labels.to(self.device)
                x_real_targets, target_labels = self.create_labels(
                    self.data_iter)
                emty_image = torch.zeros(
                    self.batch_size, 3, self.image_size, self.image_size)

                # -1 => Because of the image normalization
                x_real__visualization = torch.ones(
                    (self.batch_size+1, 3, self.image_size, self.image_size))*(-1)
                x_real__visualization[1::, :, :, :] = x_real
                x_real__visualization = x_real__visualization.to(self.device)

                x_fake_list_attention = [x_real__visualization]
                x_fake_list_reg = [x_real__visualization]
                x_fake_list_res = [x_real__visualization]
                x_fake_list_rec = [x_real__visualization]
                target_images = [emty_image]

                for idx, target_label in enumerate(target_labels):
                    attention, reg = self.G(
                        x_real, target_label.to(self.device))
                    im = self.imFromAttReg(attention, reg, x_real)
                    if not self.use_virtual:
                        reconstructed_attention, reconstructed_color_regression = self.G(
                            im, real_labels)
                        im_rec = self.imFromAttReg(
                            reconstructed_attention, reconstructed_color_regression, im)
                    else:
                        virtual_attention_mask, virtual_color_regression = self.G(
                            im, self.label_trg_virtual)
                        x_virtual = self.imFromAttReg(
                            virtual_attention_mask, virtual_color_regression, im)

                        reconstructed_virtual_attention_mask, reconstructed_virtual_color_regression = self.G(
                            x_virtual, target_label.to(self.device))
                        im_rec = self.imFromAttReg(
                            reconstructed_virtual_attention_mask, reconstructed_virtual_color_regression, x_virtual)

                    r = random.randint(0, self.batch_size-1)

                    """ Old image savings """
                    # Concatenations of the target images to the output images
                    target_concat = torch.zeros(
                        (self.batch_size+1, 3, self.image_size, self.image_size))
                    target_concat[0, :, :, :] = x_real_targets[idx][0]
                    target_concat[1::, :, :, :] = im
                    im = target_concat.to(self.device)

                    # Concatenations of the target images to the reconstructed images
                    target_concat = torch.zeros(
                        (self.batch_size+1, 3, self.image_size, self.image_size))
                    target_concat[0, :, :, :] = x_real_targets[idx][0]
                    target_concat[1::, :, :, :] = im_rec
                    im_rec = target_concat.to(self.device)

                    # Concatenations of the target images to the attentionention images
                    # Because the images are between -1 and 1
                    attention = attention.repeat(1, 3, 1, 1)*2-1

                    target_concat = torch.zeros(
                        (self.batch_size+1, 3, self.image_size, self.image_size))
                    target_concat[0, :, :, :] = x_real_targets[idx][0]
                    target_concat[1::, :, :, :] = attention
                    attention = target_concat.to(self.device)

                    # Concatenations of the color regression images to the attentionention images
                    target_concat = torch.zeros(
                        (self.batch_size+1, 3, self.image_size, self.image_size))
                    target_concat[0, :, :, :] = x_real_targets[idx][0]
                    target_concat[1::, :, :, :] = reg
                    reg = target_concat.to(self.device)

                    x_fake_list_res.append(im)
                    x_fake_list_rec.append(im_rec)
                    x_fake_list_attention.append(attention)
                    x_fake_list_reg.append(reg)

                x_concat = torch.cat(x_fake_list_res, dim=3)[
                    0:self.num_sample_targets+1, :, :, :]
                sample_path = os.path.join(
                    self.sample_dir, '{}-3-images_res.jpg'.format(self.iteration+1))
                save_image(self.denorm(x_concat.data.cpu()),
                           sample_path, nrow=1, padding=0)

                x_concat = torch.cat(x_fake_list_rec, dim=3)[
                    0:self.num_sample_targets+1, :, :, :]
                sample_path = os.path.join(
                    self.sample_dir, '{}-4-images_rec.jpg'.format(self.iteration+1))
                save_image(self.denorm(x_concat.data.cpu()),
                           sample_path, nrow=1, padding=0)

                x_concat = torch.cat(x_fake_list_attention, dim=3)[
                    0:self.num_sample_targets+1, :, :, :]
                sample_path = os.path.join(
                    self.sample_dir, '{}-1-images_attention.jpg'.format(self.iteration+1))
                save_image(self.denorm(x_concat.data.cpu()),
                           sample_path, nrow=1, padding=0)

                x_concat = torch.cat(x_fake_list_reg, dim=3)[
                    0:self.num_sample_targets+1, :, :, :]
                sample_path = os.path.join(
                    self.sample_dir, '{}-2-images_color.jpg'.format(self.iteration+1))
                save_image(self.denorm(x_concat.data.cpu()),
                           sample_path, nrow=1, padding=0)

                print('Samples saved!')

    def test(self):
        from PIL import Image
        from torchvision import transforms as T

        transform = []
        transform.append(T.ToTensor())
        transform.append(T.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)

        D_path, G_path = sorted(glob.glob(os.path.join(
            self.test_models_dir, '*.ckpt')), key=self.numericalSort)

        self.G.load_state_dict(torch.load(
            G_path, map_location=lambda storage, loc: storage))
        
        self.G = self.G.cuda()

        input_images_names = []

        with torch.no_grad():
            with open(self.test_attributes_path, 'r') as txt_file:
                csv_lines = txt_file.readlines()

                targets = torch.zeros(len(csv_lines), self.c_dim)
                input_images = torch.zeros(len(csv_lines), 3, 128, 128)

                for idx, line in enumerate(csv_lines):
                    splitted_lines = line.split(' ')
                    image_path = os.path.join(
                        self.test_images_dir, splitted_lines[0])
                    input_images[idx, :] = transform(
                        Image.open(image_path)).cuda()
                    input_images_names.append(splitted_lines[0])
                    targets[idx, :] = torch.Tensor(
                        np.array(list(map(lambda x: float(x)/5., splitted_lines[1::]))))

        test_batch_size = 7

        self.data_iter = iter(self.data_loader)
        self.x_test, _ = next(self.data_iter)
        self.x_test = self.x_test[0:test_batch_size].cuda()

        for target_idx in range(targets.size(0)):
            targets_au = targets[target_idx, :].unsqueeze(
                0).repeat(test_batch_size, 1).cuda()
            resulting_images_att, resulting_images_reg = self.G(
                self.x_test, targets_au)

            resulting_images = self.imFromAttReg(
                resulting_images_att, resulting_images_reg, self.x_test).cuda()

            save_images = -torch.ones((test_batch_size + 1)*2, 3, 128, 128).cuda()

            save_images[1:test_batch_size+1] = self.x_test
            save_images[test_batch_size+1] = input_images[target_idx]
            save_images[test_batch_size + 2:(test_batch_size + 1)*2] = resulting_images

            save_image((save_images+1)/2, os.path.join(self.test_results_dir,
                                                       input_images_names[target_idx]))

        """ Code to modify single Action Units """

        # Set data loader.
        # self.data_loader = self.data_loader

        # with torch.no_grad():
        #     for i, (self.x_real, c_org) in enumerate(self.data_loader):

        #         # Prepare input images and target domain labels.
        #         self.x_real = self.x_real.to(self.device)
        #         c_org = c_org.to(self.device)

        #         # c_trg_list = self.create_labels(self.data_loader)

        #         crit, cl_regression = self.D(self.x_real)
        #         # print(crit)
        #         print("ORIGINAL", c_org[0])
        #         print("REGRESSION", cl_regression[0])

        #         for au in range(17):
        #             alpha = np.linspace(-0.3,0.3,10)
        #             for j, a in enumerate(alpha):
        #                 new_emotion = c_org.clone()
        #                 new_emotion[:,au]=torch.clamp(new_emotion[:,au]+a, 0, 1)
        #                 attention, reg = self.G(self.x_real, new_emotion)
        #                 x_fake = self.imFromAttReg(attention, reg, self.x_real)
        #                 save_image((x_fake+1)/2, os.path.join(self.result_dir, '{}-{}-{}-images.jpg'.format(i,au,j)))

        #         if i >= 3:
        #             break
