# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(project='mmsegmentation', entity='carry-van', name='ocrnet_hr48_modify_multiscale'))
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# load_from = '/opt/ml/segmentation/baseline/mmsegmentation/load_from/ocrnet_hr48_512x512_20k_voc12aug_20200617_233932-9e82080a.pth'
# load_from = '/opt/ml/segmentation/baseline/mmsegmentation/work_dirs/9/best_mIoU_iter_40000.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
work_dir = '/opt/ml/segmentation/baseline/mmsegmentation/work_dirs/13'
