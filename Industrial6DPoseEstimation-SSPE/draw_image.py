import argparse
import warnings
warnings.filterwarnings("ignore")
import json

from darknet import Darknet
import dataset
from utils import *
from MeshPly import MeshPly

def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
    parser.add_argument("--image_size", type=int, default=544, help="The common width and height for all images")
    parser.add_argument("--p", type=str, default="model2610.weights")
    parser.add_argument("--i", type=str, default="doube-augmented/JPEGImages/1_output_0112.png")

    args = parser.parse_args()
    return args


def draw_image(opt):
    model = Darknet("yolo-pose.cfg")
    model.print_network()
    model.load_weights(opt.p)
    if torch.cuda.is_available():
        model.cuda()

    image = cv2.imread(filename=opt.i)
    original_img = np.copy(image)
    image = cv2.resize(image, (opt.image_size, opt.image_size))
    image = np.transpose(image, (2, 0, 1))
    image = image[None, :, :, :]
    image = torch.Tensor(image)

    new_image = image[0].permute(1, 2, 0).cpu().numpy()
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

    if torch.cuda.is_available():
        data = torch.from_numpy(new_image[None, :, :, :]).permute(0, 3, 1, 2).cuda()
    output = model(data).data
    all_boxes = get_region_boxes(output, 0.1, 1)
    match_thresh = 0.5
    flag = False
    bounding_box = []

    for i in range(output.size(0)):
        boxes = all_boxes[i]
        for j in range(len(boxes)):
            if boxes[j][18] > match_thresh:
                flag = True
                box_pr = boxes[j]
                corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                corners2D_pr[:, 0] = corners2D_pr[:, 0] * 640
                corners2D_pr[:, 1] = corners2D_pr[:, 1] * 480
                bounding_box.append(corners2D_pr)

        if flag == True:
            for box in bounding_box:
                for idx, point in enumerate(box):
                    x = int(float(point[0]))
                    y = int(float(point[1]))
                    cv2.circle(original_img, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(original_img, str(idx + 1), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)
                    cv2.imwrite(filename="output_image/{}".format(opt.i[-10:]), img=original_img)


if __name__ == '__main__':
    opt = get_args()
    draw_image(opt)
