import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from models.AdelaiDepth.LeReS.lib.multi_depth_model_woauxi import RelDepthModel
from models.AdelaiDepth.LeReS.lib.net_tools import load_ckpt
from models.AdelaiDepth.LeReS.lib.spvcnn_classsification import \
    SPVCNN_CLASSIFICATION
from models.AdelaiDepth.LeReS.lib.test_utils import refine_focal, refine_shift

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def reconstruct3D_from_depth(rgb, pred_depth, shift_model, focal_model):
    cam_u0 = rgb.shape[1] / 2.0
    cam_v0 = rgb.shape[0] / 2.0
    pred_depth_norm = pred_depth - pred_depth.min() + 0.5

    dmax = np.percentile(pred_depth_norm, 98)
    pred_depth_norm = pred_depth_norm / dmax

    # proposed focal length, FOV is 60', Note that 60~80' are acceptable.
    proposed_scaled_focal = (rgb.shape[0] // 2 / np.tan((60/2.0)*np.pi/180))

    # recover focal
    focal_scale_1 = refine_focal(pred_depth_norm, proposed_scaled_focal, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_1 = proposed_scaled_focal / focal_scale_1.item()

    # recover shift
    shift_1 = refine_shift(pred_depth_norm, shift_model, predicted_focal_1, cam_u0, cam_v0)
    shift_1 = shift_1 if shift_1.item() < 0.6 else torch.tensor([0.6])
    depth_scale_1 = pred_depth_norm - shift_1.item()

    # recover focal
    focal_scale_2 = refine_focal(depth_scale_1, predicted_focal_1, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_2 = predicted_focal_1 / focal_scale_2.item()

    return shift_1, predicted_focal_2, depth_scale_1

def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

class AdelaiDepth():
    def __init__(self, metric=False):
        self.weights = "./models/AdelaiDepth/weights/res101.pth"
        self.metric = metric

        self.model = RelDepthModel(backbone='resnext101')
        self.model.eval()

        self.shift_model = None
        self.focal_model = None

        if self.metric:
            # if metric is True, then recover shift and focal length
            self.shift_model = SPVCNN_CLASSIFICATION(input_channel=3, num_classes=1, cr=1.0, pres=0.01, vres=0.01)
            self.focal_model = SPVCNN_CLASSIFICATION(input_channel=5, num_classes=1, cr=1.0, pres=0.01, vres=0.01)
            self.shift_model.eval()
            self.focal_model.eval()
        
        load_ckpt(self.weights, self.model, shift_model=self.shift_model, focal_model=self.focal_model)
        self.model.to(device)

    def evaluate(self, img):
        rgb = cv2.imread(img)
        rgb_c = rgb[:, :, ::-1].copy()
        A_resize = cv2.resize(rgb_c, (448, 448))

        img_torch = scale_torch(A_resize)[None, :, :, :]
        pred_depth = self.model.inference(img_torch).cpu().numpy().squeeze()
        pred_depth = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        if self.metric:
            shift, focal_length, pred_depth = reconstruct3D_from_depth(rgb, pred_depth, self.shift_model, self.focal_model)

        # convert depth to disparity 
        pred_disp = 1.0 / pred_depth

        return pred_disp
