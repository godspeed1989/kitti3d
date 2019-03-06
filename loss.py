import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import fire
from params import para

focal_loss_ver = 0

def focal_loss(x, y, mask, eps=1e-5):
    '''Focal loss.
    Args:
        x: (tensor) sized [BatchSize, Height, Width]. predict
        y: (tensor) sized [BatchSize, Height, Width]. target {0,1}
    Return:
        (tensor) focal loss.
    '''
    alpha = 0.25
    gamma = 2

    x_t = x * (2 * y - 1) + (1 - y) # x_t = x     if label = 1
                                    # x_t = 1 -x  if label = 0

    alpha_t = torch.ones_like(x_t) * alpha
    alpha_t = alpha_t * (2 * y - 1) + (1 - y)

    if focal_loss_ver == 0:
        loss = -alpha_t * (1-x_t)**gamma * (x_t+eps).log() * mask

        return loss.sum()
    elif focal_loss_ver == 1:
        fl = (1-x_t)**gamma * (x_t+eps).log() * mask
        loss_pos = -alpha_t * fl * y
        loss_neg = -(1-alpha_t) * fl * (1-y)
        loss = loss_pos.sum() / y.sum() + loss_neg.sum() / (1-y).sum()

        return loss

def cross_entropy(x, y):
    return F.binary_cross_entropy(input=x, target=y, reduction='sum')

def smoothL1(x):
    # type: (Tensor, Tensor) -> Tensor
    with torch.no_grad():
        t = torch.abs(x)
        t = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
    return t

class CustomLoss(nn.Module):
    def __init__(self, device, num_classes=1):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

    def forward(self, preds, targets, mask=None):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          preds: (tensor)  cls_preds + reg_preds, sized[batch_size, height, width, 7]
          cls_preds: (tensor) predicted class confidences, sized [batch_size, height, width, 1].
          cls_targets: (tensor) encoded target labels, sized [batch_size, height, width, 1].
          loc_preds: (tensor) predicted target locations, sized [batch_size, height, width, 6 or 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, height, width, 6 or 8].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''

        batch_size = targets.size(0)
        image_size = targets.size(1) * targets.size(2)
        cls_preds = preds[..., :1]
        loc_preds = preds[..., 1:]

        cls_targets = targets[..., :1]
        loc_targets = targets[..., 1:]

        if mask is None:
            mask = torch.tensor(1.0, dtype=preds.dtype).to(self.device)

        ################################################################
        cls_loss = focal_loss(cls_preds, cls_targets, mask)
        ################################################################
        # cls_loss = cross_entropy(cls_preds, cls_targets)

        # ipdb.set_trace()

        ################################################################
        # reg_loss = SmoothL1Loss(loc_preds, loc_targets)
        ################################################################

        loc_loss = torch.tensor(0.0).to(self.device)
        pos_items = cls_targets.nonzero().size(0)
        if pos_items != 0:
            loc_preds_filtered = cls_targets * loc_preds
            loc_targets_filtered = cls_targets * loc_targets
            loc_loss = F.smooth_l1_loss(loc_preds_filtered, loc_targets_filtered, reduction='sum')
            # cls_targets.sum() is BAD
            loc_loss = loc_loss / (batch_size * image_size)  # Pos item is summed over all batch

        if focal_loss_ver == 0:
            cls_loss = cls_loss / (batch_size * image_size)
        alpha = 1.0
        beta = 1.0
        return alpha*cls_loss + beta*loc_loss, loc_loss.data, cls_loss.data

#######################################################################################

class GHMC_Loss:
    def __init__(self, bins=30, momentum=0.75):
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self, input, target, mask=None):
        """ Args:
        input [batch_num, class_num]:
            The prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        edges = self.edges
        mmt = self.momentum
        class_num = input.size(-1)
        input = input.reshape([-1, class_num])
        target = target.reshape([-1, class_num])
        weights = torch.zeros_like(input)

        # gradient length
        g = torch.abs(input.detach() - target)

        valid = None
        if mask is not None:
            mask = mask.reshape([-1, 1])
            valid = mask > 0.1
            tot = max(valid.float().sum().item(), 1.0)
        else:
            tot = input.size(0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1])
            if valid is not None:
                inds = inds & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            input, target, weights, reduction='sum') / tot
        return loss

class GHMR_Loss:
    def __init__(self, mu=0.02, bins=10, momentum=0.7):
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self, input, target, mask=None):
        """ Args:
        input [batch_num, 4 (* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num, 4 (* class_num)]:
            The target regression values with the same size of input.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum
        channels = input.size(-1)
        input = input.reshape([-1, channels])
        target = target.reshape([-1, channels])

        # ASL1 loss
        diff = input - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)

        valid = None
        if mask is not None:
            mask = mask.reshape([-1, 1])
            valid = mask > 0.1
            tot = max(valid.float().sum().item(), 1.0)
        else:
            tot = input.size(0)

        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1])
            if valid is not None:
                inds = inds & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss

class GHM_Loss(nn.Module):
    def __init__(self, device, num_classes=1):
        super(GHM_Loss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.ghm_cls_loss = GHMC_Loss()
        self.ghm_reg_loss = GHMR_Loss()

    def forward(self, preds, targets):
        # Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        batch_size = targets.size(0)
        image_size = targets.size(1) * targets.size(2)
        cls_preds = preds[..., 0:1]
        loc_preds = preds[..., 1:]

        cls_targets = targets[..., 0:1]
        loc_targets = targets[..., 1:]

        #cls_loss = self.ghm_cls_loss.calc(cls_preds, cls_targets, cls_targets)
        cls_loss = focal_loss(cls_preds, cls_targets) / (batch_size * image_size)
        reg_loss = self.ghm_reg_loss.calc(loc_preds, loc_targets, cls_targets)

        return cls_loss + reg_loss, reg_loss.data, cls_loss.data

#######################################################################################

class SoftmaxFocalLoss(nn.Module):
    r'''Compute multi-label loss using softmax_cross_entropy_with_logits
    focal loss for Multi-Label classification
    FL(p_t) = -alpha * (1-p_t)^{gamma} * log(p_t)
    :param
    predict {predict tensor}: format=[batch_size, channel, height, width], channel is number of class
    target {lebel tensor}: format=[batch_size, channel, height, width], channel is number of class, one-hot tensor
    reduce {bool} : true for training with multi gpu , false for one gpu
        if false , then the return is a [batch_size, 1] tensor
    size_average {bool} : if true , loss is averaged over all non-ignored pixels
                          if false , loss is averaged over batch size
    '''
    def __init__(self, alpha=0.5, gamma=3.0, size_average=True, reduce=False):
        super(SoftmaxFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, predict, target):
        target.requires_grad = False

        mask = target.ge(0.0)
        onehot = target.float()*mask.float()

        if predict.size(1) == 1:
            logits = predict + 1.e-9
        else:
            logits = nn.functional.softmax(predict, dim=1) + 1.e-9
        log_data = onehot*logits.log()
        pow_data = onehot*torch.pow(1-logits, self.gamma)
        loss = -self.alpha*pow_data*log_data*mask.float()

        if self.reduce:
            if self.size_average:
                return loss.sum()/mask.float().sum()
            else:
                return loss.sum()/mask.size(0)
        else:
            return loss.view(loss.size(0), -1).sum(1).unsqueeze(1)

class CustomLoss_ovo(nn.Module):
    def __init__(self, device='cpu', num_classes=1):
        super(CustomLoss_ovo, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.loss_cls_fn = SoftmaxFocalLoss()

    def focal_loss(self, x, y, eps=1e-5):
        '''Focal loss.
        Args:
          x: (tensor) sized [BatchSize, Height, Width].
          y: (tensor) sized [BatchSize, Height, Width].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        x_t = x * (2 * y - 1) + (1 - y) # x_t = x     if label = 1
                                        # x_t = 1 -x  if label = 0

        alpha_t = torch.ones_like(x_t) * alpha
        alpha_t = alpha_t * (2 * y - 1) + (1 - y)

        loss = -alpha_t * (1-x_t)**gamma * (x_t+eps).log()

        return loss.sum()

    def forward(self, preds, targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          preds: (tensor)  cls_preds + reg_preds, sized[batch_size, height, width, 7]
          cls_preds: (tensor) predicted class confidences, sized [batch_size, height, width, 1].
          cls_targets: (tensor) encoded target labels, sized [batch_size, height, width, 1].
          loc_preds: (tensor) predicted target locations, sized [batch_size, height, width, 6 or 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, height, width, 6 or 8].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        cls_preds = preds[..., 0]
        loc_preds = preds[..., 1:]

        cls_targets = targets[..., 0]
        loc_targets = targets[..., 1:]

        batch_size = targets.size(0)
        image_size = targets.size(1) * targets.size(2)

        #cls_loss = self.cls_loss(cls_preds.unsqueeze(1), cls_targets.unsqueeze(1))
        cls_loss = self.focal_loss(cls_preds, cls_targets)
        cls_loss = cls_loss.sum() / (batch_size * image_size)

        loc_loss = torch.tensor(0.0).to(self.device)
        pos_items = cls_targets.nonzero().size(0)
        if pos_items != 0:
            for i in range(loc_targets.shape[-1]):
                loc_preds_filtered = cls_targets * loc_preds[..., i].float()
                loc_loss += F.smooth_l1_loss(loc_preds_filtered, loc_targets[..., i], reduction='sum')

            loc_loss = loc_loss / (batch_size * image_size)

        return cls_loss+loc_loss, cls_loss, loc_loss

class MultiTaskLoss(nn.Module):
    def __init__(self, device, num_classes=1):
        super(MultiTaskLoss, self).__init__()
        num_losses = 2
        self.loss_function = CustomLoss_ovo(device=device, num_classes=num_classes)
        self.sigma = nn.Parameter(torch.ones(num_losses)/num_losses)

    def forward(self, preds, targets):
        final_loss = 0
        loss_, cls_loss, loc_loss = self.loss_function(preds, targets)
        losses = [cls_loss.cpu(), loc_loss.cpu()]
        for i, loss_i in enumerate(losses):
            final_loss += 1./(2*self.sigma[i].pow(2)) * \
                loss_i + 0.5*(self.sigma[i].pow(2)+1).log()
        return final_loss, loc_loss.data, cls_loss.data

#######################################################################################

def test():
    loss = CustomLoss(device="cpu")
    pred = torch.sigmoid(torch.randn(1, 2, 2, 3))
    label = torch.tensor([[[[1, 0.4, 0.5], [0, 0.2, 0.5]], [[0, 0.1, 0.1], [1, 0.8, 0.4]]]])
    mask = torch.ones([1, 2, 2])
    loss = loss(pred, label, mask)
    print(loss)

def test_CustomLoss_ovo():
    loss = CustomLoss_ovo()
    pred = torch.sigmoid(torch.randn(1, 2, 2, 3))
    label = torch.tensor([[[[1, 0.4, 0.5], [0, 0.2, 0.5]], [[0, 0.1, 0.1], [1, 0.8, 0.4]]]])
    loss = loss(pred, label)
    print(loss)

def test_MultiTask_loss():
    pred = torch.sigmoid(torch.randn(2, 1, 2, 3))
    label = torch.tensor([[[[1, 0.4, 0.5], [0, 0.2, 0.5]]], [[[0, 0.1, 0.1], [1, 0.8, 0.4]]]])
    N = label.size(0)
    criterion = MultiTaskLoss(device='cpu', num_classes=1)
    loss, cls_loss, loc_loss = criterion(pred, label)
    print('N %d loc %.5f cls %.5f' % (N, loc_loss, cls_loss))
    print('loss', loss)

def test_GHM_Loss():
    pred = torch.sigmoid(torch.randn(2, 1, 2, 3))
    label = torch.tensor([[[[1, 0.4, 0.5], [0, 0.2, 0.5]]], [[[0, 0.1, 0.1], [1, 0.8, 0.4]]]])
    N = label.size(0)
    criterion = GHM_Loss(device='cpu', num_classes=1)
    loss, cls_loss, loc_loss = criterion(pred, label)
    print('N %d loc %.5f cls %.5f' % (N, loc_loss, cls_loss))
    print('loss', loss)

if __name__ == "__main__":
    fire.Fire()