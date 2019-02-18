import os
import torch
import time
import ipdb
import fire
import numpy as np
from tqdm import tqdm

from params import para
from datagen import get_data_loader
from utils import get_model_name, build_model, load_config, plot_bev, plot_label_map, dict2str
from postprocess import non_max_suppression
from kitti import corners2d_to_3d, to_kitti_result_line

logf = None
def print_log(string, stdout=True, progress=None):
    if stdout:
        print(string)
    if logf is not None:
        print(string, file=logf)
        logf.flush()
    if progress is not None:
        progress.set_description_str(string)

def _get_net_input(data, device, batch_size):
    net_input = []
    if para.dense_net:
        scan = data['scan']
        net_input += [scan.to(device)]
    else:
        voxels_feature = data['voxels_feature']
        coords_pad = data['coordinates']
        net_input += [voxels_feature.to(device), coords_pad.to(device), batch_size]
    return net_input

def validate_batch(net, criterion, batch_size, val_data_loader, device):
    net.eval()
    val_loss = 0
    num_samples = 0
    for i, data in enumerate(val_data_loader):
        net_input = _get_net_input(data, device, data['cur_batch_size'])
        label_map = data['label_map'].to(device)
        if para.use_labelmap_mask:
            label_map_mask = data['label_map_mask'].to(device)
        else:
            label_map_mask = None
        predictions = net(*net_input)
        loss, loc_loss, cls_loss = criterion(predictions, label_map, label_map_mask)
        val_loss += loss.data
        num_samples += label_map.shape[0]
    return val_loss * para.batch_size / num_samples

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())


def train_net(config_name, device, val=False):
    config, learning_rate, batch_size, max_epochs = load_config(config_name)
    train_data_loader = get_data_loader('train',
        batch_size=batch_size, shuffle=True, augment=True,
        frame_range=config['frame_range'], workers=config['num_workers'])
    val_data_loader = get_data_loader('val',
        batch_size=1, shuffle=False, augment=False,
        frame_range=config['frame_range'], workers=config['num_workers'])

    net, criterion, optimizer, scheduler = build_model(config, device, train=True)

    print_log(dict2str(config))
    print_log(dict2str(para))
    if config['resume_training']:
        saved_ckpt_path = get_model_name(config['old_ckpt_name'], config, para)
        net.load_state_dict(torch.load(saved_ckpt_path, map_location=device))
        print_log("Successfully loaded trained ckpt at {}".format(saved_ckpt_path))

    net.train()
    # net.backbone.conv1.register_forward_hook(printnorm)
    # net.backbone.conv2.register_backward_hook(printgradnorm)

    start_time = time.time()
    for epoch in range(max_epochs):
        train_loss = 0
        num_samples = 0
        scheduler.step()
        print_log("Learning Rate for Epoch {} is {} ".format(epoch + 1, scheduler.get_lr()))
        pbar = tqdm(total=len(train_data_loader), initial=0, desc='train')
        for i, data in enumerate(train_data_loader):
            net_input = _get_net_input(data, device, data['cur_batch_size'])
            label_map = data['label_map'].to(device)
            if para.use_labelmap_mask:
                label_map_mask = data['label_map_mask'].to(device)
            else:
                label_map_mask = None
            optimizer.zero_grad()
            # Forward
            predictions = net(*net_input)
            # ipdb.set_trace()
            loss, loc_loss, cls_loss = criterion(predictions, label_map, label_map_mask)
            print_log('%.5f loc %.5f cls %.5f' % (loss.data, loc_loss, cls_loss), False, pbar)

            loss.backward()
            optimizer.step()

            train_loss += float(loss)
            num_samples += label_map.shape[0]
            pbar.update()
        pbar.close()
        train_loss = train_loss * batch_size/ num_samples

        print_log("Epoch {} | Time {:.3f} | Training Loss: {}".format(
            epoch + 1, time.time() - start_time, train_loss))

        if (epoch + 1) == max_epochs or (epoch + 1) % config['save_every'] == 0:
            model_path = get_model_name(para.net+'__epoch{}'.format(epoch+1), config, para)
            torch.save(net.state_dict(), model_path)
            print_log("Checkpoint saved at {}".format(model_path))
            if val:
                val_loss = validate_batch(net, criterion, batch_size, val_data_loader, device)
                print_log("Epoch {} | Validation Loss: {}".format(epoch + 1, val_loss))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print_log("Total time elapsed: {:.2f} seconds".format(elapsed_time))


def eval_one_sample(net, net_input, config, label_list=None,
                    vis=True, to_kitti_file=True, calib_dict=None):
    threshold = config['cls_threshold']
    nms_iou_threshold = config['nms_iou_threshold']
    with torch.no_grad():
        # Forward Pass
        t_start = time.time()
        if para.estimate_bh:
            pred, pred_bh = net(*net_input)
        else:
            pred = net(*net_input)
        print("Forward pass time", time.time() - t_start)

        # Select all the bounding boxes with classification score above threshold
        # [200, 175, 9]
        pred = pred[0, ...]
        cls_pred = pred[..., 0]
        activation = cls_pred > threshold

        # Compute (x, y) of the corners of selected bounding box
        num_boxes = int(activation.sum())
        if num_boxes == 0:
            print("No bounding box found")
            return []
        print('find {} bounding boxes'.format(num_boxes))

        corners = torch.zeros((num_boxes, 8))
        for i in range(1, 9):
            corners[:, i - 1] = torch.masked_select(pred[..., i], activation)
        corners = corners.view(-1, 4, 2).cpu().numpy()
        if para.estimate_bh:
            bh = torch.zeros((num_boxes, 2))
            bh[..., 0] = torch.masked_select(pred_bh[..., 0], activation)
            bh[..., 1] = torch.masked_select(pred_bh[..., 1], activation)
            bh = bh.cpu().numpy()

        scores = torch.masked_select(pred[..., 0], activation).cpu().numpy()

        # NMS
        t_start = time.time()
        selected_ids = non_max_suppression(corners, scores, nms_iou_threshold)
        corners = corners[selected_ids]
        scores = scores[selected_ids]
        if para.estimate_bh:
            bh = bh[selected_ids]
        print("Non max suppression time:", time.time() - t_start)

        if vis:
            if para.dense_net:
                input_np = net_input[0][0].cpu().numpy()
                plot_bev(input_np, predict_list=corners, label_list=label_list, window_name='result')
            else:
                voxels_feature, coords_pad = net_input[0].cpu().numpy(), net_input[1].cpu().numpy()
                channels = int((para.H2 - para.H1) / para.grid_sizeH)
                velo_processed = np.zeros((*para.input_shape, channels), dtype=np.float32)
                for i,p in enumerate(coords_pad):
                    if p[0] == 0:
                        velo_processed[p[2],p[3],p[1]] = voxels_feature[i,-1]
                plot_bev(velo_processed, predict_list=corners, label_list=label_list, window_name='result')
            plot_label_map(cls_pred.cpu().numpy())

        if to_kitti_file:
            if para.estimate_bh:
                center3d, corners3d = corners2d_to_3d(corners, bh[:,0], bh[:,1])
            else:
                center3d, corners3d = corners2d_to_3d(corners, np.array([-1.5]), np.array([.1]))
            line = to_kitti_result_line(center3d, corners3d, 'Car', scores, calib_dict)
            return line

def get_eval_net(config_name, device, config):
    # prepare model
    net, criterion = build_model(config, device, train=False)
    model_path = get_model_name(None, config, para)
    print('load {}'.format(model_path))
    net.load_state_dict(torch.load(model_path, map_location=device))
    # decode center7(cls,r1,r2,x,y,h,w) to corner9(cls,4x2)
    net.set_decode(True)
    net.eval()
    return net

def eval_net(config_name, device):
    config, _, _, _ = load_config(config_name)
    net = get_eval_net(config_name, device, config)
    loader = get_data_loader('val', batch_size=1, frame_range=config['frame_range'],
                              workers=config['num_workers'], shuffle=False, augment=False)

    for image_id, data in enumerate(loader):
        net_input = _get_net_input(data, device, 1)
        # label_list [N,4,2]
        index, boxes_3d_corners, labelmap_boxes3d_corners, labelmap_mask_boxes3d_corners, \
            calib_dict = loader.dataset.get_label(image_id)
        print('---{}---'.format(index))
        _, label_list, _ = loader.dataset.get_label_map(boxes_3d_corners, \
                labelmap_boxes3d_corners, labelmap_mask_boxes3d_corners)
        lines = eval_one_sample(net, net_input, config, label_list=label_list,
                                vis=True, to_kitti_file=True, calib_dict=calib_dict)
        for l in lines:
            print(l)

def dump_net(config_name, device, db_selection):
    config, _, _, _ = load_config(config_name)
    net = get_eval_net(config_name, device, config)

    loader = get_data_loader(db_selection, batch_size=1, frame_range=config['frame_range'],
                              workers=config['num_workers'], shuffle=False, augment=False)

    if not os.path.exists(db_selection):
        os.makedirs(db_selection, exist_ok=True)

    for image_id, data in enumerate(loader):
        net_input = _get_net_input(data, device, 1)
        if db_selection == 'test':
            index, calib_dict = loader.dataset.get_label(image_id)
        else:
            index, _, _, _, calib_dict = loader.dataset.get_label(image_id)
        lines = eval_one_sample(net, net_input, config, label_list=None,
                                vis=False, to_kitti_file=True, calib_dict=calib_dict)
        txt_file = os.path.join(db_selection, index + '.txt')
        with open(txt_file, 'w') as f:
            for l in lines:
                f.write(l)
            f.close()

'''
User functions
'''

def dump(db_selection='val'):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print('using device', device)
    name = 'config.json'
    dump_net(name, device, db_selection)

def eval():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print('using device', device)
    name = 'config.json'
    eval_net(name, device)

def train():
    global logf
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    print('using device', device)
    timestr = time.strftime("%b-%d_%H-%M-%S", time.localtime())
    logf = open('train_{}.txt'.format(timestr), 'w')
    name = 'config.json'
    train_net(name, device)
    logf.close()

if __name__ == "__main__":
    fire.Fire()
