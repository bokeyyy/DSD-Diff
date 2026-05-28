def train(self):
    for epoch in range(self.args.epochs_encoder + self.args.epochs_sr):

        self.model.train()

        self.loss.start_log()

        # 调整学习率
        self._adjust_learning_rate(epoch)

        # 训练一个 epoch
        train_loss = self._train_epoch(epoch)

        # 验证
        val_psnr, val_ssim = self._validate(epoch)

        # 保存最佳模型
        if val_psnr > self.best_psnr:
            self.best_psnr = val_psnr
            self.best_ssim = val_ssim
            self._save_model(epoch, is_best=True)

        # 保存最新模型
        self._save_model(epoch)

        # 记录日志
        self.ckp.write_log(
            f'[Epoch {epoch}] Train Loss: {train_loss:.4f}, '
            f'Val PSNR: {val_psnr:.2f}, Val SSIM: {val_ssim:.4f}'
        )


def _train_epoch(self, epoch):
    timer = util.timer()
    losses_diffusion, losses_sr = util.AverageMeter(), util.AverageMeter()

    for lr, hr in self.loader_train:
        lr, hr = lr.to(self.device), hr.to(self.device)
        self.optimizer.zero_grad()
        timer.tic()

        # 前向传播
        if epoch <= self.args.epochs_encoder:
            z, dwt2, sr = self.model((lr, hr, False))
            l_sr = self.loss(sr, hr)
            loss = l_sr
            losses_sr.update(l_sr.item())
        else:
            z, dwt2, sr = self.model((lr, hr, True))
            l_diffusion = self.loss(z, dwt2)
            l_sr = self.loss(sr, hr)
            loss = l_diffusion + l_sr * 0.01
            losses_diffusion.update(l_diffusion.item())
            losses_sr.update(l_sr.item())

        # 反向传播
        loss.backward()
        self.optimizer.step()
        timer.hold()
    # 更新学习率调度器
    self.scheduler.step()
    self.loss.step()
    if epoch <= self.args.epochs_encoder:
        if (epoch + 1) % self.args.print_every == 0:
            self.ckp.write_log(
                'Epoch: [{:03d}]'
                'Loss [sr loss: {:.3f}]\t'
                'Time [{:.1f}s]'.format(
                    epoch,
                    losses_sr.avg,
                    timer.release()
                ))
    else:
        if (epoch + 1) % self.args.print_every == 0:
            self.ckp.write_log(
                'Epoch: [{:04d}]'
                'Loss [diffusion loss:{:.6f}] [sr loss:{:.3f}]\t'
                'Time [{:.1f}s]'.format(
                    epoch,
                    losses_diffusion.avg, losses_sr.avg,
                    timer.release(),
                ))
    # 返回平均损失
    if epoch <= self.args.epochs_encoder:
        return losses_sr.avg
    else:
        return losses_diffusion.avg + losses_sr.avg * 0.01


def _validate(self, epoch):
    self.model.eval()
    eval_psnr, eval_ssim = 0, 0

    with torch.no_grad():
        for lr, hr in self.loader_val:
            lr, hr = lr.to(self.device), hr.to(self.device)

            sr = self.model((lr, hr, True))
            loss = self.loss(sr, hr)
            sr_list = torch.unbind(sr, dim=0)
            hr_list = torch.unbind(hr, dim=0)
            for i in range(len(sr_list)):
                # 量化输出
                sr = util.quantize(sr, self.args.rgb_range)
                hr = util.quantize(hr, self.args.rgb_range)

                # 计算 PSNR 和 SSIM
                psnr = util.calc_psnr(
                    sr, hr, self.args.scale, self.args.rgb_range,
                    benchmark=self.args.benchmark
                )
                ssim = util.calc_ssim(
                    sr, hr, self.args.scale,
                    benchmark=self.args.benchmark
                )
                eval_psnr += psnr
                eval_ssim += ssim

    # 计算平均 PSNR 和 SSIM
    avg_psnr = eval_psnr / (len(self.loader_val) * self.args.batch_size)
    avg_ssim = eval_ssim / (len(self.loader_val) * self.args.batch_size)

    # 记录验证结果
    self.ckp.write_log(
        f'[Epoch {epoch}] Val PSNR: {avg_psnr:.2f}, Val SSIM: {avg_ssim:.4f}'
    )

    return avg_psnr, avg_ssim
