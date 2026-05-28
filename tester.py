import os
import utility
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util
import torch.nn as nn
from model.loss_ssim import SSIMLoss


class Tester():
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_test = loader.loader_test
        self.model = my_model
        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        self.G_lossfn_weight = args.G_lossfn_weight
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.E_decay = args.E_decay

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

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

    def test(self, epoch):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))

        self.model.eval()
        timer_test = utility.timer()

        # 初始化 CUDA 时间测量事件
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0
                eval_time = 0

                degrade = util.SRMDPreprocessing(
                    self.scale[0],
                    kernel_size=self.args.blur_kernel,
                    blur_type=self.args.blur_type,
                    sig=self.args.sig,
                    lambda_1=self.args.lambda_1,
                    lambda_2=self.args.lambda_2,
                    theta=self.args.theta,
                    noise=self.args.noise
                )

                for idx_img, (input2, filename, _) in enumerate(self.loader_test):
                    input2 = input2.cuda()  # b, 1, c, h, w
                    input2 = self.crop_border(input2, scale)
                    input1, _ = degrade(input2, random=False)  # b, 1, c, h, w
                    input2 = input2[:, 0, ...]
                    input1 = input1[:, 0, ...]
                    data = [input1, input2, self.args.diff]

                    timer_test.tic()

                    # 开始计时推理过程
                    if not self.args.cpu:
                        torch.cuda.synchronize()
                        start_event.record()


                    sr = self.model(data)


                    # 结束计时
                    if not self.args.cpu:
                        end_event.record()
                        torch.cuda.synchronize()
                        elapsed_time = start_event.elapsed_time(end_event)  # 单位为毫秒 (ms)
                        eval_time += elapsed_time

                    timer_test.hold()

                    sr = utility.quantize(sr, self.args.rgb_range)
                    input2 = utility.quantize(input2, self.args.rgb_range)

                    psnr, tag = utility.calc_psnr(
                        sr, input2, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    ssim = utility.calc_ssim(
                        sr, input2, scale,
                        benchmark=self.loader_test.dataset.benchmark
                    )

                    eval_psnr += psnr
                    eval_ssim += ssim

                    # 打印单张图片的信息（包含时间）
                    print('Testing {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; Time: {:.2f}ms;'.
                          format(filename[0], psnr, ssim, elapsed_time if not self.args.cpu else 0))

                    # save results
                    if self.args.save_results:
                        sr_save_list = [sr]
                        input2_save_list = [input2]
                        input1_save_list = [input1]
                        input2_filename = filename[0] + '_hr'
                        sr_filename = filename[0] + '_sr'
                        input1_filename = filename[0] + '_lr'
                        self.ckp.save_results(input2_filename, input2_save_list, scale)
                        self.ckp.save_results(sr_filename, sr_save_list, scale)
                        self.ckp.save_results(input1_filename, input1_save_list, scale)

                # 计算平均性能并记录日志
                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} Time: {:.4f}ms'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                        eval_time / len(self.loader_test),
                    ))
        return eval_psnr / len(self.loader_test), eval_ssim / len(self.loader_test)

    def crop_border(self, input, scale):
        b, n, c, h, w = input.size()

        input = input[:, :, :, :int(h // (scale * 2 * self.args.window_size) * (scale * 2 * self.args.window_size)),
                 :int(w // (scale * 2 * self.args.window_size) * (scale * 2 * self.args.window_size))]

        return input

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs_encoder + self.args.epochs_sr

