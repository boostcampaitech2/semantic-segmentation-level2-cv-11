_base_ = [
    '../_base_/models/fcn_hr18.py',
]
model = dict(decode_head=dict(num_classes=21))
