import torch
import torch.nn as nn
import numpy as np
from functools import partial
import torch.nn.functional as F

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

class SDCDM(nn.Module):
    # classic SDCDM with Gaussian diffusion, in image space
    def __init__(self,
                 denoise,
                 condition,
                 timesteps=1000,
                 beta_schedule="linear",
                 image_size=256,
                 feats=64,
                 clip_denoised=False,
                 linear_start=0.1,
                 linear_end=0.99,
                 cosine_s=8e-3,
                 given_betas=None,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 parameterization="eps",  # all assuming fixed variance schedules
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        self.image_size = image_size
        self.channels = feats
        self.model = denoise
        self.condition = condition

        self.v_posterior = v_posterior
        self.l_simple_weight = l_simple_weight

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        # self.out = DegConFusion()

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))


        alphas_cumprod_safe = np.clip(alphas_cumprod, a_min=1e-9, a_max=1.0)


        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))


        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod_safe)))

        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod_safe)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod_safe - 1)))


        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas

        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, deg_start, con_start, t):


      
        mean_deg = extract_into_tensor(self.sqrt_alphas_cumprod, t, deg_start.shape) * deg_start
        var_deg = extract_into_tensor(1.0 - self.alphas_cumprod, t, deg_start.shape)
        log_var_deg = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, deg_start.shape)

        mean_con = extract_into_tensor(self.sqrt_alphas_cumprod, t, con_start.shape) * con_start
        var_con = extract_into_tensor(1.0 - self.alphas_cumprod, t, con_start.shape)
        log_var_con = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, con_start.shape)

        return mean_deg, mean_con, var_deg, var_con, log_var_deg, log_var_con

    def predict_start_from_noise(self, x_t, t, noise):
        # print('predict')
        # print(extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape).shape,x_t.shape)
        # print(extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape).shape,noise.shape)
        # print('out')
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        # print(extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape).shape,x_start.shape)
        # print(extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape).shape,x_t.shape)
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # print('pass')
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, deg_noisy, con_noisy, t, deg_c, con_c, clip_denoised: bool):

        # print(deg_noisy,con_noisy,t,deg_c,con_c)
        deg_out, con_out = self.model(
            deg=deg_noisy,
            con=con_noisy,
            t=t,
            deg_c=deg_c,  
            con_c=con_c 
        )
        # print('deg_out=',deg_out.shape,'con_out=', con_out.shape)

        if self.parameterization == "eps":
            deg_recon = self.predict_start_from_noise(deg_noisy, t, deg_out)
            con_recon = self.predict_start_from_noise(con_noisy, t, con_out)
        elif self.parameterization == "x0":
            deg_recon, con_recon = deg_out, con_out

        if clip_denoised:
            deg_recon.clamp_(-1., 1.)
            con_recon.clamp_(-1., 1.)

        # print('degrecon=', deg_recon.shape, 'degnoisy=', deg_noisy.shape)
        # print('conrecon=', con_recon.shape, 'connoisy=', con_noisy.shape)
        deg_mean, deg_log_var = self.q_posterior(deg_recon, deg_noisy, t)
        con_mean, con_log_var = self.q_posterior(con_recon, con_noisy, t)

        return deg_mean, con_mean, deg_log_var, con_log_var

    def p_sample(self, deg_noisy, con_noisy, t, deg_c, con_c, clip_denoised=True, repeat_noise=False):

        deg_mean, con_mean, deg_log_var, con_log_var = self.p_mean_variance(deg_noisy, con_noisy, t, deg_c, con_c,
                                                                            clip_denoised=clip_denoised)

        device = deg_mean.device
        deg_noise = noise_like(deg_mean.shape, device, repeat_noise)
        con_noise = noise_like(con_mean.shape, device, repeat_noise)

        deg_sample = deg_mean + deg_noise * (0.5 * deg_log_var).exp()
        con_sample = con_mean + con_noise * (0.5 * con_log_var).exp()

        return deg_sample, con_sample

    def q_sample(self, deg_start, con_start, t, noise=None):
        # print('deg=',deg_start.shape,'con=',con_start.shape)
        # print('noise0=',noise[0].shape,'noise1=',noise[1].shape)

        if noise is None:
            deg_noise = torch.randn_like(deg_start)
            con_noise = torch.randn_like(con_start)
        else:
            deg_noise, con_noise = noise

        # print(extract_into_tensor(self.sqrt_alphas_cumprod, t, deg_start.shape).shape,deg_start.shape)
        # print(extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, deg_start.shape).shape,deg_noise.shape)
        deg_noisy = (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, deg_start.shape) * deg_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, deg_start.shape) * deg_noise
        )
        con_noisy = (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, con_start.shape) * con_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, con_start.shape) * con_noise
        )
        return deg_noisy, con_noisy

    @torch.no_grad()
    def sample_ddim(self, img, steps, deg_cond, con_cond, eta=0.0, strength=1.0):

        device = self.betas.device
        b = img.shape[0]

        total_times = torch.linspace(0, self.num_timesteps - 1, steps=steps).long().flip(0).to(device)

        start_step_idx = int(steps * (1.0 - strength))
       
        times = total_times[start_step_idx:]

        if strength >= 1.0:
            h, w = con_cond.shape[-2:]
            deg = torch.randn(b, 256, device=device)
            con = torch.randn(b, 8, h, w, device=device)
        else:
            
            if len(times) > 0:
                t_start = torch.full((b,), times[0], device=device, dtype=torch.long)
                deg, con = self.q_sample(deg_start=deg_cond, con_start=con_cond, t=t_start)
            else:
               
                return deg_cond, con_cond
        for i, step in enumerate(times):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            current_idx_in_total = start_step_idx + i
            if current_idx_in_total < len(total_times) - 1:
                prev_t = total_times[current_idx_in_total + 1]
            else:
                prev_t = -1 
            
            deg, con = self.ddim_step(deg, con, t, prev_t, deg_cond, con_cond, eta)

        return deg, con

    def ddim_step(self, deg, con, t, prev_t, deg_c, con_c, eta):
  
        deg_pred, con_pred = self.model(deg, con, t, deg_c, con_c)

     
        alpha_bar_t = extract_into_tensor(self.alphas_cumprod, t, deg.shape)
        if prev_t >= 0:
            alpha_bar_prev = extract_into_tensor(self.alphas_cumprod, torch.full_like(t, prev_t), deg.shape)
        else:
            alpha_bar_prev = torch.tensor(1.0, device=deg.device)

        
        if self.parameterization == "x0":
            pred_x0_deg = deg_pred
            pred_x0_con = con_pred
            
            pred_x0_deg = pred_x0_deg.clamp(-3., 3.)
            pred_x0_con = pred_x0_con.clamp(-3., 3.)
           
            one_minus_at = 1 - alpha_bar_t
            one_minus_at = one_minus_at.clamp(min=0.) 
            sqrt_one_minus_at = one_minus_at.sqrt()

            safe_denom = sqrt_one_minus_at.clamp(min=1e-4)
          
            pred_eps_deg = (deg - alpha_bar_t.sqrt() * pred_x0_deg) / safe_denom
            pred_eps_con = (con - alpha_bar_t.sqrt() * pred_x0_con) / safe_denom


        if eta == 0:
            sigma = 0.
            noise_deg = 0.
            noise_con = 0.
            
            one_minus_prev = 1 - alpha_bar_prev
            one_minus_prev = one_minus_prev.clamp(min=0.) 
            dir_xt_coef = one_minus_prev.sqrt()
        else:

            sigma = 0. 
            noise_deg = 0.
            noise_con = 0.
            one_minus_prev = 1 - alpha_bar_prev
            one_minus_prev = one_minus_prev.clamp(min=0.)
            dir_xt_coef = one_minus_prev.sqrt()
            
        dir_xt_deg = dir_xt_coef * pred_eps_deg
        dir_xt_con = dir_xt_coef * pred_eps_con
        
        x_prev_deg = alpha_bar_prev.sqrt() * pred_x0_deg + dir_xt_deg + noise_deg
        x_prev_con = alpha_bar_prev.sqrt() * pred_x0_con + dir_xt_con + noise_con
        
        return x_prev_deg, x_prev_con

    def forward(self, img, deg_gt=None, con_gt=None, diffusion_strength=None):
        device = self.betas.device
        b = img.shape[0]

        deg_cond, con_cond = self.condition(img)


        if self.training:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            noise_deg = torch.randn_like(deg_gt)
            noise_con = torch.randn_like(con_gt)
            noise = (noise_deg, noise_con)
            deg_noisy, con_noisy = self.q_sample(deg_start=deg_gt, con_start=con_gt, t=t, noise=noise)
            deg_pred, con_pred = self.model(deg_noisy, con_noisy, t, deg_cond, con_cond)

            if self.parameterization == "eps":
                 loss_deg = F.mse_loss(deg_pred, noise_deg) 
                 loss_con = F.mse_loss(con_pred, noise_con) 
                 pass
            elif self.parameterization == "x0":
                 loss_deg = F.mse_loss(deg_pred, deg_gt)
                 loss_con = F.mse_loss(con_pred, con_gt)

            total_diffusion_loss = loss_deg + loss_con
            return total_diffusion_loss, deg_pred, con_pred 


        else:
            steps = 10 
            
            eff_strength = 0.4 if diffusion_strength is None else diffusion_strength
            
            deg_sample, con_sample = self.sample_ddim(
                img, 
                steps, 
                deg_cond, 
                con_cond, 
                strength=eff_strength
            )
            
            return 0.0, deg_sample, con_sample



