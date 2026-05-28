import os
import utility
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util
import torch.nn as nn
from model.loss_ssim import SSIMLoss
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from loss.vgg import VGG


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.model = my_model
        self.loss = my_loss

        self.vgg_loss = VGG(conv_index='54', rgb_range=args.rgb_range)
        self.vgg_loss.to(self.device) 

        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        # self.optimizer = utility.make_optimizer(args, self.model)
        # self.scheduler = utility.make_scheduler(args, self.optimizer)

        real_model = self.model

        if hasattr(real_model, 'module'):
            real_model = real_model.module
            # print(f"[Debug] Unwrap DataParallel -> {type(real_model)}")

        if hasattr(real_model, 'model'):
            real_model = real_model.model
            # print(f"[Debug] Unwrap model.Model -> {type(real_model)}")

        # print(f"[Debug] Final Real Model Type: {type(real_model)}")
        # print(f"[Debug] Attributes: {[attr for attr in dir(real_model) if not attr.startswith('_')]}")


        diff_params_set = set()
        if hasattr(real_model, 'netG'):
            diff_params_set.update(real_model.netG.parameters())
        if hasattr(real_model, 'condition'):
            diff_params_set.update(real_model.condition.parameters())

        if hasattr(real_model, 'dt_blocks'):
            diff_params_set.update(real_model.dt_blocks.parameters())
        if hasattr(real_model, 'f_init_conv'):
            diff_params_set.update(real_model.f_init_conv.parameters())

        diff_params = [p for p in diff_params_set if p.requires_grad]

        sr_params_set = set()
        if hasattr(real_model, 'SR'):
            sr_params_set.update(real_model.SR.parameters())
        if hasattr(real_model, 'encoder'):
            sr_params_set.update(real_model.encoder.parameters())

        sr_params = [p for p in sr_params_set if p.requires_grad]

        if len(diff_params) == 0 and len(sr_params) == 0:
            raise ValueError("error check name")


        optim_kwargs = {'weight_decay': args.weight_decay}
        optimizer_cls = torch.optim.AdamW
        if args.optimizer == 'ADAM':
            optimizer_cls = torch.optim.Adam
            optim_kwargs.update({'betas': (args.beta1, args.beta2), 'eps': args.epsilon})

        self.optimizer = optimizer_cls([
            {'params': diff_params, 'lr': 1e-4},
            {'params': sr_params, 'lr': 1e-5}
        ], **optim_kwargs)


        total_epochs = args.epochs_encoder + args.epochs_sr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs_sr,
            eta_min=1e-7
        )

        self.E_decay = args.E_decay
        # self.current_epoch = len(ckp.log)
        if args.resume > 0:
            self.current_epoch = args.resume
            print(f"[Trainer] Resume detected. Setting current_epoch to {self.current_epoch}")
        else:
        
            self.current_epoch = len(ckp.log)

        print(self.current_epoch)
        self.G_lossfn_weight = args.G_lossfn_weight

        self.E_decay = args.E_decay

        self.inference_folder1 = '/home/QLB/encoder_change_full/data/test_lr/' 
        self.inference_folder2 = '/home/QLB/encoder_change_full/data/test_hr/'  
        self.save_inference_folder = os.path.join(ckp.dir, 'inference_results')
        os.makedirs(self.save_inference_folder, exist_ok=True)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
        steps_to_fast_forward = self.current_epoch - self.args.epochs_encoder - self.args.warmup_epochs  # -warmup

        if steps_to_fast_forward > 0:
            print(
                f"[Trainer] Fast-forwarding scheduler {steps_to_fast_forward} steps to match epoch {self.current_epoch}...")
            for _ in range(steps_to_fast_forward):
                self.scheduler.step()

        # ----------------------------------------
        # define loss
        # ----------------------------------------
        G_lossfn_type = self.args.G_lossfn_type
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.args.G_lossfn_weight
        print('G_lossfn_weight')
        print(self.G_lossfn_weight)

    def load_inference_image_pairs(self, folder1, folder2, num_pairs=3):
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

     
        images1 = {}
        images2 = {}

       
        for file in sorted(os.listdir(folder1)):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                images1[file] = os.path.join(folder1, file)

    
        for file in sorted(os.listdir(folder2)):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                images2[file] = os.path.join(folder2, file)

     
        common_files = set(images1.keys()) & set(images2.keys())
        common_files = sorted(list(common_files))[:num_pairs]

        image_pairs = []
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        for filename in common_files:
          
            img1 = Image.open(images1[filename]).convert('RGB')
            img1_tensor = transform(img1).unsqueeze(0)  

          
            img2 = Image.open(images2[filename]).convert('RGB')
            img2_tensor = transform(img2).unsqueeze(0)  

            image_pairs.append((img1_tensor, img2_tensor, filename))

        print(f"Loaded {len(image_pairs)} image pairs for inference")
        return image_pairs

    def inference_and_save(self, epoch):
        # self.model.eval()


        inference_pairs = self.load_inference_image_pairs(
            self.inference_folder1,
            self.inference_folder2,
            3
        )

        total_psnr = 0
        total_ssim = 0
        num_images = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
                module.eval() 
                module.track_running_stats = False 
        with torch.no_grad():
            for i, (input1, input2, filename) in enumerate(inference_pairs):
                # input_save_folder = os.path.join(self.ckp.dir, 'input_debug')
                # os.makedirs(input_save_folder, exist_ok=True)
                # self.save_input_images(input1, input2, filename, input_save_folder, epoch)
                input1 = input1.to(self.device)
                input2 = input2.to(self.device)

             
                if epoch <= self.args.epochs_encoder:
                  
                    _, _, _, _, sr = self.model((input1, input2, False))
                else:
                 
                    _, _, _, _, sr = self.model((input1, input2, True))

              
                if isinstance(sr, (list, tuple)):
                    sr_output = sr[0] 
                else:
                    sr_output = sr

                input2_resized = input2

                # PSNR SSIM
                sr_quantized = utility.quantize(sr_output, self.args.rgb_range)
                hr_quantized = utility.quantize(input2_resized, self.args.rgb_range)

                psnr, valid = utility.calc_psnr(
                    sr_quantized, hr_quantized, self.scale[0],
                    self.args.rgb_range, benchmark=True
                )

                ssim = utility.calc_ssim(
                    sr_quantized, hr_quantized, self.scale[0],
                    benchmark=True
                )

                if valid:
                    total_psnr += psnr
                    total_ssim += ssim
                    num_images += 1


                self.save_sr_image(sr_output, epoch, filename)

                print(f'Epoch {epoch}: {filename} - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}')
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
                module.train() 
                module.track_running_stats = True
     
        if num_images > 0:
            avg_psnr = total_psnr / num_images
            avg_ssim = total_ssim / num_images
            print(f'Epoch {epoch}: Average PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}')

        
            self.ckp.write_log(
                f'[Epoch {epoch}] Inference - Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM: {avg_ssim:.4f}'
            )

        self.model.train()

    def save_input_images(self, input1, input2, filename, save_folder, epoch):
        
        name, ext = os.path.splitext(filename)

      
        input1_array = self.tensor2img(input1)
        input1_save_path = os.path.join(save_folder, f'epoch_{epoch:04d}_input1_{name}.png')
        self.save_image_array(input1_array, input1_save_path)

      
        input2_array = self.tensor2img(input2)
        input2_save_path = os.path.join(save_folder, f'epoch_{epoch:04d}_input2_{name}.png')
        self.save_image_array(input2_array, input2_save_path)

        print(f"Saved input images: {input1_save_path}, {input2_save_path}")

    def save_image_array(self, img_array, save_path):
       
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

       
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
          
            img_pil = Image.fromarray(img_array)
            img_pil.save(save_path)
        elif len(img_array.shape) == 2:
          
            img_pil = Image.fromarray(img_array, mode='L')
            img_pil.save(save_path)
        else:
       
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 1:
                    img_pil = Image.fromarray(img_array[:, :, 0], mode='L')
                else:
                    
                    img_pil = Image.fromarray(img_array[:, :, 0], mode='L')
                img_pil.save(save_path)

    def save_sr_image(self, sr_tensor, epoch, original_filename):
       
        img_array = self.tensor2img(sr_tensor)

      
        name, ext = os.path.splitext(original_filename)
        save_name = f'epoch_{epoch:04d}_sr_{name}.png'
        save_path = os.path.join(self.save_inference_folder, save_name)

        
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

       
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            
            img_pil = Image.fromarray(img_array)
            img_pil.save(save_path)
        elif len(img_array.shape) == 2:
          
            img_pil = Image.fromarray(img_array, mode='L')
            img_pil.save(save_path)
        else:
            
            if len(img_array.shape) == 3:
                
                if img_array.shape[2] == 1:
                    img_pil = Image.fromarray(img_array[:, :, 0], mode='L')
                else:
                    img_pil = Image.fromarray(img_array[:, :, 0], mode='L')
                img_pil.save(save_path)

    def tensor2img(self, tensor):
        if len(tensor.shape) == 4:
            tensor = tensor[0]  

        tensor = tensor.cpu().detach()

       
        if tensor.requires_grad:
            tensor = tensor.detach()

        img = tensor.numpy()

        if len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))

        
        if img.max() > 1.0 and img.dtype != np.uint8:
            img = img / 255.0

        return img

    def train(self):
        self.current_epoch += 1
        epoch = self.current_epoch
        train_ssim = []
        train_psnr = []

        if epoch <= self.args.epochs_encoder:
            lr = self.args.lr_encoder * (self.args.gamma_encoder ** (epoch // self.args.lr_decay_encoder))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            diff = 'off'
            self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        else:
            warmup_epochs = self.args.warmup_epochs

            stage2_epoch = epoch - self.args.epochs_encoder

            # Warmup 
            if stage2_epoch <= warmup_epochs:

                alpha = stage2_epoch / warmup_epochs

            
                diff_base_lr = self.optimizer.param_groups[0].get('initial_lr', 1e-4)
                self.optimizer.param_groups[0]['lr'] = diff_base_lr * alpha

 
                self.optimizer.param_groups[1]['lr'] = 0.0

                diff = 'on'
                lr_diff = self.optimizer.param_groups[0]['lr']
                lr_sr = 0.0
                lr = self.optimizer.param_groups[0]['lr']

            else:
                if stage2_epoch == warmup_epochs + 1:
                    print(f"[Epoch {epoch}] Warmup Done! Force resetting LRs to base values.")


                    self.optimizer.param_groups[0]['lr'] = 1e-4  
                    self.optimizer.param_groups[1]['lr'] = 1e-5  

                    if hasattr(self, 'scheduler'):
                        self.scheduler.base_lrs = [1e-4, 1e-5]

                    print(f"LRs reset to: Diff=1e-4, SR=1e-5. Scheduler synced.")

                lr_diff = self.optimizer.param_groups[0]['lr']
                lr_sr = self.optimizer.param_groups[1]['lr']
                lr = self.optimizer.param_groups[0]['lr']
            # lr = self.args.lr_sr * (self.args.gamma_sr ** ((epoch - self.args.epochs_encoder) // self.args.lr_decay_sr))
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = lr
            # lr_diff = self.optimizer.param_groups[0]['lr']
            # lr_sr = self.optimizer.param_groups[1]['lr']

            self.ckp.write_log(
                '[Epoch {}]\tLR Diff: {:.2e} | LR SR: {:.2e}'.format(epoch, Decimal(lr_diff), Decimal(lr_sr)))

        # self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        degrade = util.SRMDPreprocessing(
            self.scale[0],
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )

        timer = utility.timer()
        losses_diffusion, losses_sr, losses_vgg = utility.AverageMeter(), utility.AverageMeter(), utility.AverageMeter()

        i = 0
        for batch, (input2, _, idx_scale) in enumerate(self.loader_train):
            input2 = input2.cuda()  # b, n, c, h, w
            input1, b_kernels = degrade(input2)  # bn, c, h, w
            input1 = input1[:, 0, ...]
            input2 = input2[:, 0, ...]
            data = [input1, input2, diff]

            self.optimizer.zero_grad()

            timer.tic()
            # forward
            ## train stage1
            if epoch <= self.args.epochs_encoder:

                _, deg_diff, spa_diff, deg, spa, sr = self.model(data)
                # print(input2[1].max(), input2[1].min())
                l_sr = self.loss(sr, input2)
                loss = l_sr
                losses_sr.update(l_sr.item())
                if epoch % 50 == 0:
                    for i in range(len(sr)):
                        _sr = utility.quantize(sr[i].unsqueeze(0), self.args.rgb_range)
                        _input2 = utility.quantize(input2[i].unsqueeze(0), self.args.rgb_range)

                        psnr, tag = utility.calc_psnr(
                            _sr, _input2, 4, self.args.rgb_range,
                            benchmark='True'
                        )
                        ssim = utility.calc_ssim(
                            _sr, _input2, 4,
                            benchmark='True'
                        )
                        if tag == 1:
                            train_ssim.append(ssim)
                            train_psnr.append(psnr)
            ## train stage2
            else:
                # print('stage2')
                l_diffusion, deg_diff, spa_diff, deg, spa, sr = self.model(data)
                # ld_diffusion = self.loss(deg_diff, deg)
                # lc_diffusion = self.loss(spa_diff, spa)
                # l_diffusion = ld_diffusion * 1000 + lc_diffusion
                l_sr_pixel = self.loss(sr, input2)
                l_sr_vgg = self.vgg_loss(sr, input2)
                loss = (l_diffusion * 500) + l_sr_pixel + (l_sr_vgg)
                # l_sr = self.loss(sr, input2)
                # loss = l_diffusion * 2000 + l_sr
                # loss_deg.update(ld_diffusion.item())
                # loss_spa.update(lc_diffusion.item())
                losses_diffusion.update(l_diffusion.item())
                losses_sr.update(l_sr_pixel.item())
                losses_vgg.update(l_sr_vgg.item())
                if epoch % 50 == 0:
                    for i in range(len(sr)):
                        _sr = utility.quantize(sr[i].unsqueeze(0), self.args.rgb_range)
                        _input2 = utility.quantize(input2[i].unsqueeze(0), self.args.rgb_range)

                        psnr, tag = utility.calc_psnr(
                            _sr, _input2, 4, self.args.rgb_range,
                            benchmark='True'
                        )
                        ssim = utility.calc_ssim(
                            _sr, _input2, 4,
                            benchmark='True'
                        )
                        if tag == 1:
                            train_ssim.append(ssim)
                            train_psnr.append(psnr)
            # if epoch == 800:
            #     print(f"deg_diff sum: {deg_diff.sum().item():.6f}")
            #     print(f"spa_diff sum: {spa_diff.sum().item():.6f}")
            #     print(f"deg sum: {deg.sum().item():.6f}")
            #     print(f"spa sum: {spa.sum().item():.6f}")

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) 
            self.optimizer.step()
            timer.hold()

            if epoch <= self.args.epochs_encoder:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                        'Loss [sr loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_sr.avg,
                            timer.release()
                        ))
            else:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:04d}][{:04d}/{:04d}]\n'
                        'Loss [diffusion loss:{:.6f}] [sr loss:{:.3f}] [vgg loss:{:.3f}] '
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_diffusion.avg, losses_sr.avg, losses_vgg.avg,
                            timer.release(),
                        ))
        if epoch > self.args.epochs_encoder:
            stage2_epoch = epoch - self.args.epochs_encoder

            if stage2_epoch > self.args.warmup_epochs:
                self.scheduler.step()
        self.loss.step()
        if epoch % 50 == 0:
            print(f'train_ssim = {sum(train_ssim) / len(train_ssim)}, train_psnr = {sum(train_psnr) / len(train_psnr)}')
        # save model
        if epoch >= 0 and (epoch + 1) % 10 == 0:
            target = self.model.get_model()
            model_dict = target.state_dict()
            keys = list(model_dict.keys())
            # for key in keys:
            #     if 'encoder' in key:
            #         del model_dict[key]
            torch.save(
                model_dict,
                os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch + 1))
            )
        return epoch

   
    def crop_border(self, img_hr, scale):
        b, n, c, h, w = img_hr.size()
        img_hr = img_hr[:, :, :, :int(h // scale * scale), :int(w // scale * scale)]
        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            # epoch = self.scheduler.last_epoch + 1
            # return epoch >= self.args.epochs_encoder + self.args.epochs_sr
            return self.current_epoch >= self.args.epochs_encoder + self.args.epochs_sr