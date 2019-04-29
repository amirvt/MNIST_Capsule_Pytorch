import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

BATCH_SIZE = 128
device = torch.device('cuda')
NUM_EPOCHS = 10
torch.manual_seed(1)


class MinMaxNormalize(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())


transform = transforms.Compose([
    transforms.ToTensor(),
    MinMaxNormalize()
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)


# %%
class ConvLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(**kwargs)

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(**kwargs) for _ in range(num_capsules)
        ])

    def forward(self, xx):
        xs = [squash(c(xx).view(xx.shape[0], -1)) for c in self.convs]
        xs = torch.stack(xs, dim=-1)
        return xs


class Reconstruct(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, output, target):
        tensors = torch.stack([output[i][target[i]] for i in range(output.shape[0])], 0)
        images = self.decoder(tensors)
        return images


class DigitCaps(nn.Module):

    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(32 * 6 * 6, 8, 10, 16))

    def forward(self, x):
        u_hat = torch.einsum('bri,rico->brco', [x, self.W])
        b = torch.zeros(*u_hat.shape[:-1], device=device)  # brc
        for i in range(3):
            c = F.softmax(b, -2)
            s = torch.einsum('brc,brco->bco', [c, u_hat])
            v = squash(s)
            if i != 2:
                b = b + torch.einsum('brco,bco->brc', [u_hat, v])
            else:
                return v


class CapsNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(in_channels=1, out_channels=256, kernel_size=9)
        self.primary = PrimaryCaps(8, in_channels=256, out_channels=32, kernel_size=9, stride=2)
        self.digits = DigitCaps()
        self.reconstruct = Reconstruct()

    def forward(self, x):
        return self.digits(self.primary(self.conv1(x)))


def norm_squared(tensor):
    norm = tensor.pow(2).sum(-1, keepdim=True)
    return norm


def squash(tensor):
    norm = norm_squared(tensor)
    return (norm / (norm + 1)) * (tensor / norm.sqrt())


def margin_loss(v, target_one_hot):
    v_mag = norm_squared(v).sqrt()
    zero = torch.zeros(1, device=device)
    left = torch.max((0.9 - v_mag.squeeze(-1)), zero) ** 2
    right = torch.max((v_mag.squeeze(-1) - .1), zero) ** 2

    loss = target_one_hot * left + 0.5 * (1.0 - target_one_hot) * right

    loss = loss.sum()

    return loss


def one_hot(y):
    return torch.zeros(y.shape[0], 10, device=device).scatter_(-1, y.unsqueeze(-1), 1)


model = CapsNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
reconstruction_criterion = nn.MSELoss(reduction='sum')

for i_epoch in range(1, NUM_EPOCHS):
    model.train()
    pbar = tqdm(train_loader)
    loss_total = 0
    acc_total = 0
    total = 0
    for batch_idx, (data, target) in (enumerate(pbar)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target_one_hot = one_hot(target)

        loss = margin_loss(output, target_one_hot)

        images = model.reconstruct(output, target)
        images = images.view(*data.shape)
        r_loss = reconstruction_criterion(images, data)
        loss += 5e-4 * r_loss

        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        acc_total += (norm_squared(output).squeeze(-1).max(-1)[1] == target).sum().item()
        total += target.shape[0]

        pbar.set_description(
            '[ TRN ][ {:02d} / {:02d} ][ LSS: {:.3f} ACC: {:.3f} ]'.format(i_epoch, NUM_EPOCHS, loss_total / total,
                                                                           acc_total / total))

    model.eval()
    pbar = tqdm(test_loader)
    loss_total = 0
    acc_total = 0
    total = 0
    for batch_idx, (data, target) in (enumerate(pbar)):
        data, target = data.to(device), target.to(device)

        output = model(data)
        target_one_hot = one_hot(target)
        loss = margin_loss(output, target_one_hot)

        images = model.reconstruct(output, target)
        images = images.view(*data.shape)
        r_loss = reconstruction_criterion(images, data)
        loss += 5e-4 * r_loss

        loss_total += loss.item()
        acc_total += (norm_squared(output).squeeze(-1).max(-1)[1] == target).sum().item()
        total += target.shape[0]

        pbar.set_description(
            '[ TST ][ {:02d} / {:02d} ][ LSS: {:.3f} ACC: {:.3f} ]'.format(i_epoch, NUM_EPOCHS, loss_total / total,
                                                                           acc_total / total))
