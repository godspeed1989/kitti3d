import torch
import time
import ipdb
import fire
from tqdm import tqdm

from params import para
from datagen import get_data_loader
from model import build_model
from utils import get_model_name, load_config, plot_bev, plot_label_map, dict2str
from postprocess import non_max_suppression
from kitti import to_kitti_result_line

logf = None
def print_log(string, stdout=True, progress=None):
    if stdout:
        print(string)
    if logf is not None:
        print(string, file=logf)
        logf.flush()
    if progress is not None:
        progress.set_description_str(string)

def validate_batch(net, criterion, batch_size, val_data_loader, device):
    net.eval()
    val_loss = 0
    num_samples = 0
    for i, data in enumerate(val_data_loader):
        input, label_map = data
        input = input.to(device)
        label_map = label_map.to(device)
        predictions = net(input)
        loss, loc_loss, cls_loss = criterion(predictions, label_map)
        val_loss += float(loss)
        num_samples += label_map.shape[0]
    return val_loss * batch_size / num_samples

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
    train_data_loader, val_data_loader = get_data_loader(
        batch_size=batch_size, input_channels=config['input_channels'],
        use_npy=config['use_npy'], frame_range=config['frame_range'], workers=config['num_workers'])
    net, criterion, optimizer, scheduler = build_model(config, device, train=True)

    print_log(dict2str(config))
    if config['resume_training']:
        saved_ckpt_path = get_model_name(config['old_ckpt_name'], config['input_channels'], para.box_code_len)
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
        for i, (input, label_map) in enumerate(train_data_loader):
            input = input.to(device)
            label_map = label_map.to(device)
            optimizer.zero_grad()
            # Forward
            predictions = net(input)
            # ipdb.set_trace()
            loss, loc_loss, cls_loss = criterion(predictions, label_map)
            print_log('loc_loss %.5f cls_loss %.5f' % (loc_loss, cls_loss), False, pbar)
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
            model_path = get_model_name(config['name']+'__epoch{}'.format(epoch+1), config['input_channels'], para.box_code_len)
            torch.save(net.state_dict(), model_path)
            print_log("Checkpoint saved at {}".format(model_path))
            if val:
                val_loss = validate_batch(net, criterion, batch_size, val_data_loader, device)
                print_log("Epoch {} | Validation Loss: {}".format(epoch + 1, val_loss))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print_log("Total time elapsed: {:.2f} seconds".format(elapsed_time))


def eval_one_sample(net, input, label_map, label_list, config,
                    vis=True, to_kitti_file=True):
    threshold = config['cls_threshold']
    nms_iou_threshold = config['nms_iou_threshold']
    with torch.no_grad():
        # Forward Pass
        t_start = time.time()
        pred = net(input.unsqueeze(0)).squeeze_(0)
        print("Forward pass time", time.time() - t_start)

        # Select all the bounding boxes with classification score above threshold
        # [200, 175, 9]
        cls_pred = pred[..., 0]
        activation = cls_pred > threshold

        # Compute (x, y) of the corners of selected bounding box
        num_boxes = int(activation.sum())
        if num_boxes == 0:
            print("No bounding box found")
            return
        print('find {} bounding boxes'.format(num_boxes))

        corners = torch.zeros((num_boxes, 8))
        for i in range(1, 9):
            corners[:, i - 1] = torch.masked_select(pred[..., i], activation)
        corners = corners.view(-1, 4, 2).cpu().numpy()

        scores = torch.masked_select(pred[..., 0], activation).cpu().numpy()

        # NMS
        t_start = time.time()
        selected_ids = non_max_suppression(corners, scores, nms_iou_threshold)
        corners = corners[selected_ids]
        scores = scores[selected_ids]
        print("Non max suppression time:", time.time() - t_start)

        if vis:
            input_np = input.cpu().numpy()
            plot_bev(input_np, label_list, window_name='GT')
            plot_bev(input_np, corners, window_name='Prediction')
            plot_label_map(cls_pred.cpu().numpy())

        if to_kitti_file:
            line = to_kitti_result_line(corners)
            print(line)


def eval_net(config_name, device, net=None, all_sample=False):
    config, _, _, _ = load_config(config_name)

    # prepare model
    if net is None:
        net, criterion = build_model(config, device, train=False)
        model_path = get_model_name(config['name'] + '__epoch100', config['input_channels'], para.box_code_len)
        print('load {}'.format(model_path))
        net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        pass
    # decode center7(cls,r1,r2,x,y,h,w) to corner9(cls,4x2)
    net.set_decode(True)
    _, loader = get_data_loader(batch_size=1, input_channels=config['input_channels'],
                                use_npy=config['use_npy'], frame_range=config['frame_range'])
    net.eval()

    if all_sample is not True:
        image_id = 5
        input, label_map = loader.dataset[image_id]
        input = input.to(device)
        label_map = label_map.to(device)
        # label_list [N,4,2]
        boxes_3d_corners, labelmap_boxes3d_corners = loader.dataset.get_label(image_id)
        label_map_unnorm, label_list = loader.dataset.get_label_map(boxes_3d_corners, labelmap_boxes3d_corners)
        eval_one_sample(net, input, label_map, label_list, config, vis=True, to_kitti_file=False)
    else:
        for i, data in enumerate(loader):
            input, label_map = data
            input = input.to(device)
            label_map = label_map.to(device)
            # label_list [N,4,2]
            boxes_3d_corners = loader.dataset.get_label(image_id)
            label_map_unnorm, label_list = loader.dataset.get_label_map(boxes_3d_corners)
            eval_one_sample(net, input, label_map, label_list, config, vis=False, to_kitti_file=True)

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
