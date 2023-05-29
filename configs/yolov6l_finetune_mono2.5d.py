# YOLOv6l model
model = dict(
    type='YOLOv6l',
    pretrained='weights/yolov6l.pt',
    depth_multiple=1.0,  
    width_multiple=1.0,
    bins=2,
    overlap=0.1,
    backbone=dict(
        type='CSPBepBackbone',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        csp_e=float(1)/2,
        fuse_P2=True,
        ),
    neck=dict(
        type='CSPRepBiFPANNeck',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
        csp_e=float(1)/2,
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=3,
        anchors_init=[[10,13, 19,19, 33,23], 
                      [30,61, 59,59, 59,119], 
                      [116,90, 185,185, 373,326]],
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        atss_warmup_epoch=0,
        iou_type='giou',
        use_dfl=False,
        reg_max=0, #if use_dfl is False, please set reg_max to 0
        distill_weight={
            'class': 2.0,
            'dfl': 1.0,
        },
        loss_weight={
            'class': 1.0,
            'iou': 2.5,
            'dfl': 0.5,
            'dims': 0.5,
            'conf': 1.0,
            'orint': 3.0
        }
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.0032,
    lrf=0.12,
    momentum=0.843,
    weight_decay=0.00036,
    warmup_epochs=2.0,
    warmup_momentum=0.5,
    warmup_bias_lr=0.05
)

data_aug = dict(
    hsv_h=0.0138,
    hsv_s=0.664,
    hsv_v=0.464,
    degrees=0.373,
    translate=0.245,
    scale=0.898,
    shear=0.602,
    flipud=0.00856,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.243,
)
training_mode = "conv_silu" 
# use normal conv to speed up training and further improve accuracy.

eval_params=dict(
    do_coco_metric=False,
    do_pr_metric=True,
    plot_curve=True,
    plot_confusion_matrix=True
)