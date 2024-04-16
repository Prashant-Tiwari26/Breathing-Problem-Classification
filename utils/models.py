import torch
import torch.nn.functional as F
from torchvision.models.regnet import RegNet_Y_3_2GF_Weights, regnet_y_3_2gf

def vae_loss(x, x_recon, mu, logvar):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def vae_loss_torch(x, x_recon):
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = F.kl_div(x_recon, x, reduction='batchmean')
    return BCE + KLD

class VariationalAutoencoderBase(torch.nn.Module):
    def __init__(self, dim_list: list) -> None:
        super().__init__()
        self.encoder = torch.nn.ModuleList()
        for i in range(1, len(dim_list)):
            self.encoder.append(torch.nn.Linear(in_features=dim_list[i-1], out_features=dim_list[i]))
            self.encoder.append(torch.nn.BatchNorm1d(dim_list[i]))
            self.encoder.append(torch.nn.SELU())

        self.decoder = torch.nn.ModuleList()
        dim_list.reverse()
        self.decoder.append(torch.nn.Linear(in_features=dim_list[0]//2, out_features=dim_list[1]))
        self.decoder.append(torch.nn.BatchNorm1d(dim_list[1]))
        self.decoder.append(torch.nn.SELU())
        for i in range(2, len(dim_list)):
            self.decoder.append(torch.nn.Linear(in_features=dim_list[i-1], out_features=dim_list[i]))
            self.decoder.append(torch.nn.BatchNorm1d(dim_list[i]))
            self.decoder.append(torch.nn.SELU())

        self.apply(self._init_weights)
        print("Number of Parameters = {:,}".format(self._get_num_params()))
        
    def _reparametrize(self, mean, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        mean, logvar = torch.chunk(x, 2, -1)
        z = self._reparametrize(mean, logvar)
        for layer in self.decoder:
            z = layer(z)
        y = torch.sigmoid(z)
        return y, mean, logvar

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _get_num_params(self):
        return sum(p.numel() for p in self.parameters())

class RegNetMultimodalBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = torch.nn.Sequential(*list(regnet_y_3_2gf(weights=RegNet_Y_3_2GF_Weights.IMAGENET1K_V2).children())[:-1])
        self.ann = torch.nn.Sequential(
            torch.nn.Linear(17, 50),
            torch.nn.BatchNorm1d(50),
            torch.nn.SELU(),
            torch.nn.Linear(50, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.SELU(),
            torch.nn.Linear(100, 150),
            torch.nn.BatchNorm1d(150),
            torch.nn.SELU(),
        )
        self.classify = torch.nn.Linear(1662, 22)

        self.apply(self._init_weights)
        print("Number of Parameters = {:,}".format(self._get_num_params()))

    def forward(self, img, x):
        o1 = self.cnn(img).squeeze(dim=[-1,-2])
        o2 = self.ann(x)
        output = torch.cat([o1, o2], 1)
        output = self.classify(output)
        return output
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _get_num_params(self):
        return sum(p.numel() for p in self.parameters())