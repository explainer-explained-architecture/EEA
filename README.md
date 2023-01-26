# PyTorch Implementation of Explainer-Explained Architecture (EEA) for Vision Transformer

We present the ‘explainer-explained architecture’ (EEA) - a novel method for producing explanations by learning explanation masks. The EEA introduces an explainer-explained framework, in which the explainer learns to explain and justify the explained model’s predictions.
The EEA framework, presented in this paper for the first time, is a general explainability framework that can be applied to many types of supervised models in different domains. In this work, we demonstrate EEA’s ability to produce explanations for ViT models, where it significantly outperforms state-of-the-art methods on multiple explanations and segmentation tests.

<img src="images\2_classes_vis_github.png" alt="2_classes_vis_github" width="250" height="200" align:center/>

<img src="images\single_object_vis_github.png" alt="single_object_vis_github" width="200" height="350" align:center />


## Reproducing results on ViT-Base & ViT-Small - Pertubations Metrics
---
### Loading Checkpoints:
- Download `checkpoints.zip` from https://drive.google.com/file/d/1syOvmnXFgMsIgu-10LNhm0pHDs2oo1gm/
- unzip classifier.zip -d ./checkpoints/ (after unzipping, the checkpointes should be in the corresponding folders based on the backbone's type (`vit_base` / `vit_small`))

These checkpoints are important for reproducing the results. All explanation metrics can be calculated using the mask files created during the EEA procedure.

### Evaluations

#### EEA

```python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/seg_classification/run_seg_cls_opt.py --RUN-BASE-MODEL False --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --train-model-by-target-gt-class True
```

#### pEEA

```python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/seg_classification/run_seg_cls_opt.py --RUN-BASE-MODEL True --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --train-model-by-target-gt-class True
```
### Pretraining Phase - pEEA model

```python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/seg_classification/run_seg_cls.py --enable-checkpointing True --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --mask-loss-mul 50 --train-model-by-target-gt-class True --n-epochs 30 --train-n-label-sample 1
```



## Reproducing results on ViT-Base & ViT-Small - Segmentation Results

---
### Download the segmentaion datasets:
- Download imagenet_dataset [Link to download dataset](http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat)
- Download the COCO_Val2017 [Link to download dataset](https://cocodataset.org/#download)
- Download Pascal_val_2012 [Link to download dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

- Move all datasets to ./data/

### pEEA

```python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/segmentation_eval/seg_stage_a.py --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --dataset-type imagenet
```

### EEA

```python
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/segmentation_eval/seg_stage_b.py --explainer-model-name vit_base_224 --explainee-model-name vit_base_224 --dataset-type imagenet
```

** The dataset can be chosen by the parameter of `--dataset-type` from `imagenet`, `coco`, `voc`