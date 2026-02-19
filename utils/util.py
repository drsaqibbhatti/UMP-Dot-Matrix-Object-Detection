import copy
import math
import random
import time
import numpy
import torch
import torchvision
from torch.nn.functional import cross_entropy, one_hot
import os




def save_checkpoint(state, model, checkpoint_dir, epoch):
    # Remove the previous checkpoint only if the epoch is greater than 0
    if epoch > 0:
        # Remove the previous checkpoint
        previous_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch - 1}.pth')
        if os.path.exists(previous_checkpoint):
            try:
                os.remove(previous_checkpoint)
                #print(f"Previous checkpoint {previous_checkpoint} removed.")
            except Exception as e:
                print(f"Error removing previous checkpoint: {e}")
        else:
            print(f"No previous checkpoint found at {previous_checkpoint}")
        
    
    # Save the new checkpoint
    filename = f'checkpoint_epoch_{epoch}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)

    # Script the model and save it as part of the checkpointing process
    # Check if the model is wrapped in DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # Unwrap DataParallel to get the original model



    #print(f"Checkpoint saved at {checkpoint_path}")



def load_checkpoint(checkpoint_path, model, optimizer):
    # Load the optimizer state and other metadata
    checkpoint = torch.load(checkpoint_path,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {epoch}")
    return epoch

def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





def make_anchors(x, strides, offset=0.5):
    """
    Generate anchors from features
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p

def box_iou(box1, box2):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = intersection / (area1 + area2 - intersection)
    box1 = box1.T
    box2 = box2.T

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1[:, None] + area2 - intersection)


def wh2xy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_threshold=0.1, iou_threshold=0.45):
    nc = prediction.shape[1] - 4  # nc is number of classes
    xc = prediction[:, 4:4 + nc].amax(1) > conf_threshold  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_det = 300 # the maximum number of boxes to keep after NMS
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    start = time.time()
    outputs = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for index, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (box, conf, cls)
        box, cls = x.split((4, nc), 1)
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = wh2xy(box)
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]
        # Check shape
        if not x.shape[0]:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        i = i[:max_det]  # limit detections
        outputs[index] = x[i]
        if (time.time() - start) > 0.5 + 0.05 * prediction.shape[0]:
            print(f'WARNING ⚠️ NMS time limit {0.5 + 0.05 * prediction.shape[0]:.3f}s exceeded')
            break  # time limit exceeded

    return outputs


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap



def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


class EMA:

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()



class ComputeLoss:
    def __init__(self, model, params):
        super().__init__()
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device  # get model device

        m = model.head  # Head() module
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device
        self.params = params

        # task aligned assigner
        self.top_k = 10
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9

        self.bs = 1
        self.num_max_boxes = 0
        # DFL Loss params
        self.dfl_ch = m.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)

    def __call__(self, outputs, targets):
    
        x = outputs[1] if isinstance(outputs, tuple) else outputs
        
        output = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2) 

        pred_output, pred_scores = output.split((4 * self.dfl_ch, self.nc), 1)
            
        pred_output = pred_output.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            
            
        size = torch.tensor(x[0].shape[2:], dtype=pred_scores.dtype, device=self.device)
        size = size * self.stride[0]

        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)
        
        # Process targets as a list of dictionaries
        batch_size = pred_scores.shape[0]
        max_num_boxes = max(len(t["boxes"]) for t in targets)  # Find the max number of boxes across the batch

        # Initialize tensor to hold gt (ground truth) data
        gt = torch.zeros(batch_size, max_num_boxes, 5, device=self.device)

        for i, target in enumerate(targets):
            num_boxes = len(target["boxes"])
            if num_boxes > 0:
                # Concatenate class labels and bounding boxes
                gt[i, :num_boxes, 0] = target["labels"]  # Assign class labels
                gt[i, :num_boxes, 1:5] = target["boxes"]  # Assign bounding boxes

            
        # Convert bounding boxes from (cx, cy, w, h) format if necessary
        # print("GT Boxes before conversion:", gt[..., 1:5])
        # print("GT Boxes after conversion:", wh2xy(gt[..., 1:5]))
        gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))
        # print("GT Boxes after wh2xy and multiply size conversion:", gt[..., 1:5])
        # Split ground truth data into class labels and bounding boxes
        gt_labels, gt_bboxes = gt.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        # boxes
        b, a, c = pred_output.shape
        pred_bboxes = pred_output.view(b, a, 4, c // 4).softmax(3)
        pred_bboxes = pred_bboxes.matmul(self.project.type(pred_bboxes.dtype))

        a, b = torch.split(pred_bboxes, 2, -1)
        pred_bboxes = torch.cat((anchor_points - a, anchor_points + b), -1)

        scores = pred_scores.detach().sigmoid()
        bboxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
        target_bboxes, target_scores, fg_mask = self.assign(scores, bboxes,
                                                            gt_labels, gt_bboxes, mask_gt,
                                                            anchor_points * stride_tensor)

        target_bboxes /= stride_tensor
        target_scores_sum = target_scores.sum()

        # cls loss
        loss_cls = self.bce(pred_scores, target_scores.to(pred_scores.dtype))
        loss_cls = loss_cls.sum() / target_scores_sum

        # box loss
        loss_box = torch.zeros(1, device=self.device)
        loss_dfl = torch.zeros(1, device=self.device)
        if fg_mask.sum():
            # IoU loss
            weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_box = self.iou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            # #######################################for IoU ##################
            # pred_bboxes_filtered= pred_bboxes[fg_mask]
            # target_bboxes_filtered=  target_bboxes[fg_mask]
            # ####################################################
            loss_box = ((1.0 - loss_box) * weight).sum() / target_scores_sum
            # DFL loss
            a, b = torch.split(target_bboxes, 2, -1)
            target_lt_rb = torch.cat((anchor_points - a, b - anchor_points), -1)
            target_lt_rb = target_lt_rb.clamp(0, self.dfl_ch - 1.01)  # distance (left_top, right_bottom)
            loss_dfl = self.df_loss(pred_output[fg_mask].view(-1, self.dfl_ch), target_lt_rb[fg_mask])
            loss_dfl = (loss_dfl * weight).sum() / target_scores_sum


        loss_cls *= self.params['cls']
        loss_box *= self.params['box']
        loss_dfl *= self.params['dfl']
        


        return loss_cls + loss_box + loss_dfl # loss(cls, box, dfl)

    @torch.no_grad()
    def assign(self, pred_scores, pred_bboxes, true_labels, true_bboxes, true_mask, anchors):
        """
        Task-aligned One-stage Object Detection assigner
        """
        self.bs = pred_scores.size(0)
        self.num_max_boxes = true_bboxes.size(1)

        if self.num_max_boxes == 0:
            device = true_bboxes.device
            return (torch.full_like(pred_scores[..., 0], self.nc).to(device),
                    torch.zeros_like(pred_bboxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device))
    
        i = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)
        i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)
        i[1] = true_labels.long().squeeze(-1)

        overlaps = self.iou(true_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1))
        overlaps = overlaps.squeeze(3).clamp(0)


        align_metric = pred_scores[i[0], :, i[1]].pow(self.alpha) * overlaps.pow(self.beta)      
        
        bs, n_boxes, _ = true_bboxes.shape
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)
        mask_in_gts = bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9)
        metrics = align_metric * mask_in_gts
        top_k_mask = true_mask.repeat([1, 1, self.top_k]).bool()
        num_anchors = metrics.shape[-1]
        top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=True)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0)
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2)
        # filter invalid boxes
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k = is_in_top_k.to(metrics.dtype)
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * true_mask

        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.num_max_boxes, 1])
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes)
            is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(-2)
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        # assigned target labels, (b, 1)
        batch_index = torch.arange(end=self.bs,
                                   dtype=torch.int64,
                                   device=true_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes
        target_labels = true_labels.long().flatten()[target_gt_idx]

        # assigned target boxes
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = one_hot(target_labels, self.nc)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2)
        norm_align_metric = norm_align_metric.unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool()

    @staticmethod
    def df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        l_loss = cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
        r_loss = cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
        return (l_loss * wl + r_loss * wr).mean(-1, keepdim=True)

    @staticmethod
    def iou(box1, box2, eps=1e-7):
        # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        area1 = b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)
        area2 = b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        intersection = area1.clamp(0) * area2.clamp(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - intersection + eps

        # IoU
        iou = intersection / union
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        # center dist ** 2
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (rho2 / c2 + v * alpha)  # CIoU
    


class ComputeIoU:
    def __init__(self, model):
        super().__init__()
        if hasattr(model, 'module'):
            model = model.module

        device = next(model.parameters()).device  # get model device

        m = model.head  # Head() module
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device
        self.dfl_ch = m.dfl.ch
        
        # task aligned assigner
        self.top_k = 10
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9
        
        self.bs = 1
        self.num_max_boxes = 0
        self.dfl_ch = m.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)


    def __call__(self, outputs, targets):
    
        x = outputs[1] if isinstance(outputs, tuple) else outputs
        
        output = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2) 

        pred_output, pred_scores = output.split((4 * self.dfl_ch, self.nc), 1)
            
        pred_output = pred_output.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
            
            
        size = torch.tensor(x[0].shape[2:], dtype=pred_scores.dtype, device=self.device)
        size = size * self.stride[0]

        anchor_points, stride_tensor = make_anchors(x, self.stride, 0.5)
        
        # Process targets as a list of dictionaries
        batch_size = pred_scores.shape[0]
        max_num_boxes = max(len(t["boxes"]) for t in targets)  # Find the max number of boxes across the batch

        # Initialize tensor to hold gt (ground truth) data
        gt = torch.zeros(batch_size, max_num_boxes, 5, device=self.device)

        for i, target in enumerate(targets):
            num_boxes = len(target["boxes"])
            if num_boxes > 0:
                # Concatenate class labels and bounding boxes
                gt[i, :num_boxes, 0] = target["labels"]  # Assign class labels
                gt[i, :num_boxes, 1:5] = target["boxes"]  # Assign bounding boxes

            
        # Convert bounding boxes from (cx, cy, w, h) format if necessary
        # print("GT Boxes before conversion:", gt[..., 1:5])
        # print("GT Boxes after conversion:", wh2xy(gt[..., 1:5]))
        gt[..., 1:5] = wh2xy(gt[..., 1:5].mul_(size[[1, 0, 1, 0]]))
        # print("GT Boxes after wh2xy and multiply size conversion:", gt[..., 1:5])
        # Split ground truth data into class labels and bounding boxes
        gt_labels, gt_bboxes = gt.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        # boxes
        b, a, c = pred_output.shape
        pred_bboxes = pred_output.view(b, a, 4, c // 4).softmax(3)
        pred_bboxes = pred_bboxes.matmul(self.project.type(pred_bboxes.dtype))

        a, b = torch.split(pred_bboxes, 2, -1)
        pred_bboxes = torch.cat((anchor_points - a, anchor_points + b), -1)
        scores = pred_scores.detach().sigmoid()
        bboxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
        target_bboxes, target_scores, fg_mask = self.assign(scores, bboxes,
                                                            gt_labels, gt_bboxes, mask_gt,
                                                            anchor_points * stride_tensor)

        target_bboxes /= stride_tensor

        if fg_mask.sum():
            pred_bboxes_filtered= pred_bboxes[fg_mask]
            target_bboxes_filtered=  target_bboxes[fg_mask]

        # print('predicted_boxes',pred_bboxes_filtered)
        # print("target boxes",target_bboxes_filtered)
        return pred_bboxes_filtered, target_bboxes_filtered 

    @torch.no_grad()
    def assign(self, pred_scores, pred_bboxes, true_labels, true_bboxes, true_mask, anchors):
        """
        Task-aligned One-stage Object Detection assigner
        """
        self.bs = pred_scores.size(0)
        self.num_max_boxes = true_bboxes.size(1)

        if self.num_max_boxes == 0:
            device = true_bboxes.device
            return (torch.full_like(pred_scores[..., 0], self.nc).to(device),
                    torch.zeros_like(pred_bboxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device))
    
        i = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)
        i[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)
        i[1] = true_labels.long().squeeze(-1)

        overlaps = self.iou(true_bboxes.unsqueeze(2), pred_bboxes.unsqueeze(1))
        overlaps = overlaps.squeeze(3).clamp(0)


        align_metric = pred_scores[i[0], :, i[1]].pow(self.alpha) * overlaps.pow(self.beta)      
        
        bs, n_boxes, _ = true_bboxes.shape
        lt, rb = true_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)
        mask_in_gts = bbox_deltas.view(bs, n_boxes, anchors.shape[0], -1).amin(3).gt_(1e-9)
        metrics = align_metric * mask_in_gts
        top_k_mask = true_mask.repeat([1, 1, self.top_k]).bool()
        num_anchors = metrics.shape[-1]
        top_k_metrics, top_k_indices = torch.topk(metrics, self.top_k, dim=-1, largest=True)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])
        top_k_indices = torch.where(top_k_mask, top_k_indices, 0)
        is_in_top_k = one_hot(top_k_indices, num_anchors).sum(-2)
        # filter invalid boxes
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k = is_in_top_k.to(metrics.dtype)
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_top_k * mask_in_gts * true_mask

        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.num_max_boxes, 1])
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = one_hot(max_overlaps_idx, self.num_max_boxes)
            is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask = mask_pos.sum(-2)
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)

        # assigned target labels, (b, 1)
        batch_index = torch.arange(end=self.bs,
                                   dtype=torch.int64,
                                   device=true_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_index * self.num_max_boxes
        target_labels = true_labels.long().flatten()[target_gt_idx]

        # assigned target boxes
        target_bboxes = true_bboxes.view(-1, 4)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = one_hot(target_labels, self.nc)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2)
        norm_align_metric = norm_align_metric.unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_bboxes, target_scores, fg_mask.bool()

    @staticmethod
    def iou(box1, box2, eps=1e-7):
        """Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)"""
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        area1 = w1 * h1
        area2 = w2 * h2
        intersection = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                       (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        union = area1 + area2 - intersection + eps

        return intersection / union




#########################################################################################

import cv2
import numpy as np

from utils.util import box_iou  # Ensure this is imported

# def visualize_predictions(image, gt_boxes, pred_boxes, gt_labels=None, pred_labels=None, pred_scores=None):
#     """
#     Visualize ground truth and predicted boxes with class-specific colors.

#     Args:
#         image (torch.Tensor): Image tensor (C, H, W).
#         gt_boxes (np.ndarray or Tensor): GT boxes [x_min, y_min, x_max, y_max].
#         pred_boxes (np.ndarray or Tensor): Pred boxes [x_min, y_min, x_max, y_max].
#         gt_labels (list or Tensor): GT class indices.
#         pred_labels (list or Tensor): Predicted class indices.
#         pred_scores (list or Tensor): Predicted confidence scores.
#     """

#     # Define class-color mapping (extend if needed)
#     CLASS_COLORS = {
#         0: (0, 255, 0),    # Green for class 0
#         1: (0, 255, 255),  # Yellow for class 1
#         2: (255, 0, 0),    # Blue for class 2
#     }

#     # Convert image to NumPy
#     if isinstance(image, torch.Tensor):
#         image = image.permute(1, 2, 0).cpu().numpy().astype(np.float32)

#     image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

#     # Convert boxes to tensor if needed
#     gt_boxes = torch.tensor(gt_boxes) if not isinstance(gt_boxes, torch.Tensor) else gt_boxes
#     pred_boxes = torch.tensor(pred_boxes) if not isinstance(pred_boxes, torch.Tensor) else pred_boxes

#     if gt_labels is not None:
#         gt_labels = gt_labels.tolist() if isinstance(gt_labels, torch.Tensor) else gt_labels
#     if pred_labels is not None:
#         pred_labels = pred_labels.tolist() if isinstance(pred_labels, torch.Tensor) else pred_labels
#     if pred_scores is not None:
#         pred_scores = pred_scores.tolist() if isinstance(pred_scores, torch.Tensor) else pred_scores

#     used_pred_indices = set()

#     # Draw ground truth boxes
#     for i, gt_box in enumerate(gt_boxes):
#         label = gt_labels[i] if gt_labels else 0
#         color = CLASS_COLORS.get(label, (255, 255, 255))  # default white

#         x1, y1, x2, y2 = map(int, gt_box)
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(image, f"GT: {label}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

#         # Match to best prediction (IoU-based)
#         if pred_boxes is not None and len(pred_boxes) > 0:
#             ious = box_iou(gt_box.unsqueeze(0), pred_boxes)[0]
#             ious[torch.tensor(list(used_pred_indices), dtype=torch.long)] = -1 # mask used predictions

#             best_idx = torch.argmax(ious).item()
#             if ious[best_idx] > 0:
#                 used_pred_indices.add(best_idx)
#                 pred_box = pred_boxes[best_idx]
#                 x1p, y1p, x2p, y2p = map(int, pred_box)
#                 pred_label = pred_labels[best_idx] if pred_labels else "?"
#                 score = pred_scores[best_idx] if pred_scores else 0.0
#                 pred_color = CLASS_COLORS.get(pred_label, (0, 0, 255))  # fallback red

#                 cv2.rectangle(image, (x1p, y1p), (x2p, y2p), pred_color, 2)
#                 cv2.putText(image, f"P:{pred_label} {score:.2f}", (x1p, y1p - 8),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)

#     cv2.imshow("Predictions vs GT", image)
#     cv2.waitKey(10)


def visualize_predictions(image, gt_boxes, pred_boxes, gt_labels=None, pred_labels=None, pred_scores=None):
    """
    Visualize ground truth and the predicted box with the maximum IoU using OpenCV.
    Args:
        image (torch.Tensor): Input image tensor in [C, H, W] format.
        gt_boxes (torch.Tensor or np.ndarray): Ground truth boxes in [x_min, y_min, x_max, y_max] format.
        pred_boxes (torch.Tensor or np.ndarray): Predicted boxes in [x_min, y_min, x_max, y_max] format.
        gt_labels (torch.Tensor or list): Ground truth labels (optional).
        pred_labels (torch.Tensor or list): Predicted labels (optional).
        pred_scores (torch.Tensor or list): Confidence scores for predicted boxes (optional).
    """
    # **Normalize and Convert Image for OpenCV**
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # Convert [C, H, W] to [H, W, C]
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # Normalize to 0-255 and cast to uint8

    # Draw Ground Truth Boxes (Green)
    for i, box in enumerate(gt_boxes):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()  # Convert tensor to NumPy array
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green for GT
        if gt_labels is not None:
            cv2.putText(image, f"GT: {gt_labels[i]}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for i, gt_box in enumerate(gt_boxes):
            # Ensure gt_box is 2D
            if isinstance(gt_box, torch.Tensor):
                gt_box = gt_box.unsqueeze(0)  # Add batch dimension if it's a single box
            elif isinstance(gt_box, np.ndarray):
                gt_box = np.expand_dims(gt_box, axis=0)  # Handle NumPy arrays similarly

            # Ensure pred_boxes is a PyTorch tensor
            if not isinstance(pred_boxes, torch.Tensor):
                pred_boxes = torch.tensor(pred_boxes)

            # Compute IoU
            ious = box_iou(torch.tensor(gt_box), pred_boxes)  # IoU of this GT box with all predictions
            max_iou_idx = torch.argmax(ious, dim=1).item()  # Index of the box with the highest IoU

            if ious[0, max_iou_idx] > 0:  # If there's at least one overlapping prediction
                max_pred_box = pred_boxes[max_iou_idx]
                if isinstance(max_pred_box, torch.Tensor):
                    max_pred_box = max_pred_box.cpu().numpy()  # Convert tensor to NumPy array
                x_min, y_min, x_max, y_max = map(int, max_pred_box)

                # Draw the box with maximum IoU (Red)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                if pred_labels is not None and pred_scores is not None:
                    cv2.putText(image, f"Pred: {pred_labels[max_iou_idx]} {pred_scores[max_iou_idx]:.6f}",
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    # **Display Image**
    cv2.imshow("Ground Truth and Predictions", image)
    cv2.waitKey(10)  # Adjust delay as needed


################################################

def SegIOU(target, prediction):
    prediction = np.where(prediction > 0.5, 1, 0)
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    summation = np.sum(union)

    if summation == 0:
        return 0

    iou_score = np.sum(intersection) / summation
    return iou_score


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        Dice_BCE = dice_loss

        return Dice_BCE
    

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, gamma=2, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gamma = gamma


    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer


        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky
    
    
def pixel_iou_from_nms(outputs, targets, width, height, visualize=False):
    vis_gt_boxes = []
    vis_pred_boxes = []
    vis_pred_labels = []
    vis_pred_scores = []
    per_class_totals = {}
    per_class_counts = {}

    for target in targets:
        boxes = target['boxes']
        x_center, y_center, box_width, box_height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height
        x_min = x_center - (box_width / 2)
        y_min = y_center - (box_height / 2)
        x_max = x_center + (box_width / 2)
        y_max = y_center + (box_height / 2)
        target['boxes'] = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    outputs_nms = non_max_suppression(outputs, conf_threshold=0.01, iou_threshold=0.5)

    total_iou = 0.0
    total_boxes = 0

    for j, output in enumerate(outputs_nms):
        if output is None or output.size(0) == 0:
            continue

        target_box = targets[j]['boxes']
        target_class = targets[j]['labels'].long()
        predicted_boxes = output[:, :4]
        predicted_classes = output[:, 5].long()
        predicted_scores = output[:, 4]

        for cls in torch.unique(target_class):
            gt_mask = target_class == cls
            pred_mask = predicted_classes == cls
            gt_boxes_cls = target_box[gt_mask]
            pred_boxes_cls = predicted_boxes[pred_mask]

            if gt_boxes_cls.size(0) == 0 or pred_boxes_cls.size(0) == 0:
                continue

            iou = box_iou(gt_boxes_cls, pred_boxes_cls)
            max_ious_per_gt, max_indices = torch.max(iou, dim=1)

            total_iou += max_ious_per_gt.sum().item()
            total_boxes += gt_boxes_cls.size(0)

            cls_id = cls.item()
            per_class_totals[cls_id] = per_class_totals.get(cls_id, 0.0) + max_ious_per_gt.sum().item()
            per_class_counts[cls_id] = per_class_counts.get(cls_id, 0) + gt_boxes_cls.size(0)

            if visualize:
                
                matched_pred_indices = set(max_indices.tolist())
                all_pred_indices = set(range(pred_boxes_cls.size(0)))
                
                
                for gt_idx, pred_idx in enumerate(max_indices):
                    vis_gt_boxes.append(gt_boxes_cls[gt_idx].cpu().numpy())
                    vis_pred_boxes.append(pred_boxes_cls[pred_idx].cpu().numpy())
                    vis_pred_labels.append(cls.item())
                    vis_pred_scores.append(predicted_scores[pred_mask][pred_idx].cpu().item())

                unmatched_pred_indices = all_pred_indices - matched_pred_indices
                for idx in unmatched_pred_indices:
                    vis_pred_boxes.append(pred_boxes_cls[idx].cpu().numpy())
                    vis_pred_labels.append(cls.item())
                    vis_pred_scores.append(predicted_scores[pred_mask][idx].cpu().item())



    mean_iou = total_iou / total_boxes if total_boxes > 0 else 0

    if visualize:
        return mean_iou, vis_gt_boxes, vis_pred_boxes, vis_pred_labels, vis_pred_scores, per_class_totals, per_class_counts
    else:
        return mean_iou, per_class_totals, per_class_counts

    
# def compute_mean_iou_from_nms(outputs, targets, width, height, visualize=False):

#     vis_gt_boxes = []
#     vis_pred_boxes = []
#     vis_pred_labels = []
#     vis_pred_scores = []

#     for target in targets:
#         boxes = target['boxes']
#         x_center, y_center, box_width, box_height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#         x_center *= width
#         y_center *= height
#         box_width *= width
#         box_height *= height
#         x_min = x_center - (box_width / 2)
#         y_min = y_center - (box_height / 2)
#         x_max = x_center + (box_width / 2)
#         y_max = y_center + (box_height / 2)
#         target['boxes'] = torch.stack([x_min, y_min, x_max, y_max], dim=1)

#     outputs_nms = non_max_suppression(outputs, conf_threshold=0.001, iou_threshold=0.5)

#     total_iou = 0.0
#     total_boxes = 0

#     for j, output in enumerate(outputs_nms):
#         if output is None or output.size(0) == 0:
#             continue

#         target_box = targets[j]['boxes']
#         target_class = targets[j]['labels'].long()
#         predicted_boxes = output[:, :4]
#         predicted_classes = output[:, 5].long()
#         predicted_scores = output[:, 4]

#         for cls in torch.unique(target_class):
#             gt_mask = target_class == cls
#             pred_mask = predicted_classes == cls
#             gt_boxes_cls = target_box[gt_mask]
#             pred_boxes_cls = predicted_boxes[pred_mask]

#             if gt_boxes_cls.size(0) == 0 or pred_boxes_cls.size(0) == 0:
#                 continue

#             iou = box_iou(gt_boxes_cls, pred_boxes_cls)
#             max_ious_per_gt, max_indices = torch.max(iou, dim=1)

#             total_iou += max_ious_per_gt.sum().item()
#             total_boxes += gt_boxes_cls.size(0)

#             if visualize:
#                 for gt_idx, pred_idx in enumerate(max_indices):
#                     vis_gt_boxes.append(gt_boxes_cls[gt_idx].cpu().numpy())
#                     vis_pred_boxes.append(pred_boxes_cls[pred_idx].cpu().numpy())
#                     vis_pred_labels.append(cls.item())
#                     vis_pred_scores.append(predicted_scores[pred_mask][pred_idx].cpu().item())

#     mean_iou = total_iou / total_boxes if total_boxes > 0 else 0

#     if visualize:
#         return mean_iou, vis_gt_boxes, vis_pred_boxes, vis_pred_labels, vis_pred_scores
#     else:
#         return mean_iou



def compute_iou(box1, box2):
    """
    box1: Tensor of shape (N, 4) or (1, 4)
    box2: Tensor of shape (M, 4)
    Returns IoU tensor of shape (N, M)
    """
    # Ensure 2D tensors
    if box1.ndim == 1:
        box1 = box1.unsqueeze(0)
    if box2.ndim == 1:
        box2 = box2.unsqueeze(0)

    # Extract corners
    x1 = torch.max(box1[:, None, 0], box2[:, 0])  # (N, M)
    y1 = torch.max(box1[:, None, 1], box2[:, 1])
    x2 = torch.min(box1[:, None, 2], box2[:, 2])
    y2 = torch.min(box1[:, None, 3], box2[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # (N,)
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # (M,)

    union_area = area1[:, None] + area2 - inter_area  # (N, M)

    return inter_area / union_area.clamp(min=1e-8)  # Avoid division by zero




def iou_from_nms_auc(outputs, targets, width, height, object_score_th, visualize=False):
    vis_gt_boxes = []
    vis_pred_boxes = []
    vis_pred_labels = []
    vis_pred_scores = []
    per_class_totals = {}
    per_class_counts = {}
    per_class_tp = {}
    per_class_fp = {}
    per_class_fn = {}

    # Convert target boxes from normalized center format to pixel x1y1x2y2 format
    for target in targets:
        boxes = target['boxes']
        x_center, y_center, box_width, box_height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height
        x_min = x_center - (box_width / 2)
        y_min = y_center - (box_height / 2)
        x_max = x_center + (box_width / 2)
        y_max = y_center + (box_height / 2)
        target['boxes'] = torch.stack([x_min, y_min, x_max, y_max], dim=1)

    outputs_nms = non_max_suppression(outputs, conf_threshold=object_score_th, iou_threshold=0.5)

    total_iou = 0.0
    total_boxes = 0
    iou_threshold = 0.5

    for j, output in enumerate(outputs_nms):
        if output is None or output.size(0) == 0:
            continue

        target_box = targets[j]['boxes']
        target_class = targets[j]['labels'].long()
        predicted_boxes = output[:, :4]
        predicted_classes = output[:, 5].long()
        predicted_scores = output[:, 4]

        for cls in torch.unique(target_class):
            cls_id = cls.item()
            gt_mask = target_class == cls
            pred_mask = predicted_classes == cls
            gt_boxes_cls = target_box[gt_mask]
            pred_boxes_cls = predicted_boxes[pred_mask]
            pred_scores_cls = predicted_scores[pred_mask]

            if gt_boxes_cls.size(0) == 0 and pred_boxes_cls.size(0) == 0:
                continue

            if gt_boxes_cls.size(0) == 0:
                per_class_fp[cls_id] = per_class_fp.get(cls_id, 0) + pred_boxes_cls.size(0)
                continue
            if pred_boxes_cls.size(0) == 0:
                per_class_fn[cls_id] = per_class_fn.get(cls_id, 0) + gt_boxes_cls.size(0)
                continue

            iou = box_iou(gt_boxes_cls, pred_boxes_cls)
            max_ious_per_gt, max_indices = torch.max(iou, dim=1)

            total_iou += max_ious_per_gt.sum().item()
            total_boxes += gt_boxes_cls.size(0)

            per_class_totals[cls_id] = per_class_totals.get(cls_id, 0.0) + max_ious_per_gt.sum().item()
            per_class_counts[cls_id] = per_class_counts.get(cls_id, 0) + gt_boxes_cls.size(0)

            # Track TP/FP/FN
            matched_gt = torch.zeros(gt_boxes_cls.size(0), dtype=torch.bool)
            matched_pred = torch.zeros(pred_boxes_cls.size(0), dtype=torch.bool)

            for gt_idx in range(gt_boxes_cls.size(0)):
                pred_idx = max_indices[gt_idx]
                if iou[gt_idx, pred_idx] >= iou_threshold and not matched_pred[pred_idx]:
                    matched_gt[gt_idx] = True
                    matched_pred[pred_idx] = True

            tp = matched_gt.sum().item()
            fn = (~matched_gt).sum().item()
            fp = (~matched_pred).sum().item()

            per_class_tp[cls_id] = per_class_tp.get(cls_id, 0) + tp
            per_class_fp[cls_id] = per_class_fp.get(cls_id, 0) + fp
            per_class_fn[cls_id] = per_class_fn.get(cls_id, 0) + fn

            # Visualization
            if visualize:
                matched_pred_indices = set(max_indices[matched_gt].tolist())
                all_pred_indices = set(range(pred_boxes_cls.size(0)))

                for gt_idx, pred_idx in enumerate(max_indices):
                    if matched_gt[gt_idx]:
                        vis_gt_boxes.append(gt_boxes_cls[gt_idx].cpu().numpy())
                        vis_pred_boxes.append(pred_boxes_cls[pred_idx].cpu().numpy())
                        vis_pred_labels.append(cls_id)
                        vis_pred_scores.append(pred_scores_cls[pred_idx].cpu().item())

                unmatched_pred_indices = all_pred_indices - matched_pred_indices
                for idx in unmatched_pred_indices:
                    vis_pred_boxes.append(pred_boxes_cls[idx].cpu().numpy())
                    vis_pred_labels.append(cls_id)
                    vis_pred_scores.append(pred_scores_cls[idx].cpu().item())

    mean_iou = total_iou / total_boxes if total_boxes > 0 else 0.0

    if visualize:
        return mean_iou, vis_gt_boxes, vis_pred_boxes, vis_pred_labels, vis_pred_scores, per_class_totals, per_class_counts, per_class_tp, per_class_fp, per_class_fn
    else:
        return mean_iou, per_class_totals, per_class_counts, per_class_tp, per_class_fp, per_class_fn
