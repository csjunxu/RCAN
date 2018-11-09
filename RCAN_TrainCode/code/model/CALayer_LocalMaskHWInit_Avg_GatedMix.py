# feature maps in each layer share a common mask
class CALayer_LocalMaskHWInit_Avg_GatedMix(nn.Module):
    def __init__(self, channel, mask_height, mask_width, reduction=16):
        super(CALayer_LocalMaskHWInit_Avg_GatedMix, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # global maximum pooling: feature --> point
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv2d = nn.Conv2d(1, 1, (mask_height, mask_width), stride=(mask_height, mask_width))
        self.sigmoid = nn.Sigmoid()
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self._initialize_weights()

    def forward(self, x):
        # pdb.set_trace()
        bs, ch, h, w = x.size()
        x_t = x.view(bs*ch, 1, h, w)
        xc = self.conv2d(x_t)
        # average pooling
        xc_avg = self.avg_pool(xc)
        xc_avg_t = xc_avg.view(bs, ch, 1, 1)
        alpha = self.sigmoid(xc_avg_t)
        # maximum pooling
        # xc_max = self.max_pool(xc)
        # xc_max_t = xc_max.view(bs, ch, 1, 1)
        # beta = self.sigmoid(xc_max_t)
        # print(alpha)
        y = alpha*self.avg_pool(x)+(1-alpha)*self.max_pool(x)
        # y = beta * self.avg_pool(x) + (1 - beta) * self.max_pool(x)
        # y = alpha*self.avg_pool(x)+(1-alpha)*self.max_pool(x)+ \
            # +beta * self.avg_pool(x)+(1-beta)*self.max_pool(x)
        # y = alpha*self.avg_pool(x)+(1-alpha)*self.max_pool(x)+\
            # (1-beta)*self.avg_pool(x)+beta*self.max_pool(x)
        y = self.conv_du(y)
        return x * y

    # initial the weights only for the mask
    def _initialize_weights(self):
        init.orthogonal_(self.conv2d.weight, init.calculate_gain('relu'))
                                                                                                                                                                                                 72,1          20%
