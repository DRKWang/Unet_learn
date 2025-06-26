from tqdm import tqdm
from funcs import *

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, noise_strength):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.noise_strength = noise_strength
    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)

        noise = torch.randn_like(x_0)
        noise = self.noise_strength * noise

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss

class GaussianDiffusionSampler(nn.Module):
    # this is for the reverse process.
    def __init__(self, model, beta_1, beta_T, T, noise_strength):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.noise_strength = noise_strength

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        # 1/sqrt(a_1)(x_t - (1-a_t)/sqrt(1-a_t) e)
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        # this function is used to generate the mean and variance of random variables,
        # mean =  1/sqrt(a_t)(x_t - (1-a_t)/sqrt(1-a_t) e(x_t, t)),
        # var = sigma_t
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        backward from x_T to x_0
        """

        x_t = x_T
        timestep_list = list(reversed(range(self.T)))
        for time_step in tqdm(timestep_list):
            # print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            # x_p moment's mean and variance.
            mean, var= self.p_mean_variance(x_t=x_t, t=t)

            # no noise when t == 0
            if time_step > 0:
                # noise = self.noise_level * torch.randn_like(x_t)
                noise = torch.randn_like(x_t)
                noise = self.noise_strength * noise
            else:
                noise = 0
        
            x_t = mean + torch.sqrt(var)  * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)