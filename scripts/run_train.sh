#!/bin/bash

python scripts/train.py --multirun \
  method=contrastive \
  model.contrastive.z_dim=10 \
  data.static_noise=true \
  train.contrastive.lr=0.0003,0.0005,0.0007,0.001 \
  model.contrastive.temperature=0.1,0.25,0.5,1.0,2.0,3.0,5.0,10.0,20.0,30.0 \
  train.contrastive.batch_size=256,512,1024,2048 \
  train.steps_phase1=100
