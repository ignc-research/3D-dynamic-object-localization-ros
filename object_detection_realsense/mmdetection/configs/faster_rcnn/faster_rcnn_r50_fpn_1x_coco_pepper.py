_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)
                )
            )

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('object',)
data = dict(
    train=dict(
        img_prefix='/home/daerion/Programs/mmdetection_models/pepper/',
        classes=classes,
        ann_file='/home/daerion/Programs/mmdetection_models/pepper/train.json'),
    val=dict(
        img_prefix='/home/daerion/Programs/mmdetection_models/pepper/',
        classes=classes,
        ann_file='/home/daerion/Programs/mmdetection_models/pepper/val.json'),
    test=dict(
        img_prefix='/home/daerion/Programs/mmdetection_models/pepper/',
        classes=classes,
        ann_file='/home/daerion/Programs/mmdetection_models/pepper/test.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = '/home/daerion/Programs/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
