#%%
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


class Logger():
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag,value,step)

    def image_summary(self, tag, image, step):
        x = vutils.make_grid(image, normalize=True, scale_each=True)
        self.writer.add_image(tag, x, step)

#%%
