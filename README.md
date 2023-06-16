# PyTorch Implementation of Explainer-Explained Architecture (EEA) for Vision Models

This paper introduces the Explainer-Explained Architecture (EEA), a novel architecture for post-hoc explanations for vision models. EEA incorporates an _explainer_ model designed to generate explanation maps that emphasize the most crucial regions that justify the predictions of the model being _explained_. The training regimen for the explainer involves an initial pre-training stage, followed by a per-instance finetuning stage. The optimization during both stages employs a unique configuration in which the explained model's prediction for a masked input is compared to its original prediction for the unmasked input. This approach enables a novel counterfactual objective, aiming to anticipate the model's result using masked versions of the input image. Notably, EEA is model-agnostic and showcases its capacity to yield explanations for both Transformer-based and convolutional models. Our evaluations show that EEA substantially surpasses the current state-of-the-art in explainability for vision Transformers while delivering competitive results in explaining convolutional models.

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
