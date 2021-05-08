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
    parser.add_argument("--i", type=str, default="input_video.mp4")

    args = parser.parse_args()
    return args


def draw_image(opt):
    model = Darknet("yolo-pose.cfg")
    model.load_weights(opt.p)
    if torch.cuda.is_available():
        model.cuda()

    input_video = cv2.VideoCapture(opt.i)
    output_video = cv2.VideoWriter("output_video_result.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 5,
                                   (1280, 720))
    count = 0
    while input_video.isOpened():
        count += 1
        print(count)
        flag, image = input_video.read()
        output_image = np.copy(image)

        if flag:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            break

        image = cv2.resize(image, (opt.image_size, opt.image_size))
        image = np.transpose(image, (2, 0, 1))
        image = image[None, :, :, :]
        image = torch.Tensor(image)
        #print(image.shape)
        image = torch.rot90(image,3,(2,3))
        #print(image.shape)
        new_image = image[0].permute(1, 2, 0).cpu().numpy()
        if torch.cuda.is_available():
            data = torch.from_numpy(new_image[None, :, :, :]).permute(0, 3, 1, 2).cuda()
        output = model(data).data

        all_boxes = get_region_boxes(output, 0.1, 1)
        match_thresh = 0.7

        for i in range(output.size(0)):
            boxes = all_boxes[i]

            for j in range(len(boxes)):
                flag = False
                if boxes[j][18] > match_thresh:
                    flag = True
                    box_pr = boxes[j]

                    corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                    corners2D_pr[:, 0] = corners2D_pr[:, 0] * 1280
                    corners2D_pr[:, 1] = corners2D_pr[:, 1] * 720
                if flag == True:
                    for idx, point in enumerate(corners2D_pr):
                        x = int(float(point[0]))
                        y = int(float(point[1]))
                        cv2.circle(output_image, (x, y), 3, (0, 255, 0), -1)
                        cv2.putText(output_image, str(idx + 1), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 1)
                        #cv2.imwrite(filename="output_image/{}".format(opt.input[-10:]), img=output_image)
        output_video.write(output_image)
    output_video.release()

if __name__ == '__main__':
    opt = get_args()
    draw_image(opt)
