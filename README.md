# Consistency Regularization for Semi-supervised Learning with PyTorch
This repositrory includes consistency regularization algorithms for semi-supervised learning:
- Pi-Model
- Pseudo-label
- Mean Teacher
- Virtual Adversarial Training
- Interpolation Consistency Training
- Unsupervised Data Augmentation
- FixMatch (with RandAugment)

Training and evaluation setting follow Oliver+ 2018 and FixMatch.

# Requirements
- Python >= 3.7
- PyTorch >= 1.0
- torchvision >= 0.4
- NumPy
- sklearn (optional)

sklean is used for moon_data_exp.py (two moons dataset experiment)

# Usage
One can use ```sh ./scripts/DATASET_NAME/ALGORITHM.sh /PATH/TO/OUTPUT_DIR NUM_LABELS```,
for example, to reproduce fixmatch in CIFAR-10 with 250 labels results, run

```
sh ./scripts/fixmatch-setup/cifar10/fixmatch.sh ./results/cifar10-fixmatch-250labeles 250
```

The scripts in ```scripts/fixmatch-setup``` are for training and evaluating a model with the FixMatch setting,
and the scripts in ```scripst/realistic-evaluation-setup``` are for training and evaluating a model with the Oliver+ 2018 setting.

If yor would like to train a model with own setting, please see ```parser.py```.

NOTE: ```train_test.py``` evaluates a model performance as median of last [1, 10, 20, 50] checkpoint accuracies (FixMatch setting),
and ```train_val_test.py``` evaluates the test accuracy of the best model on validation data (Oliver+ 2018 setting).

# Performance
WIP
||Oliver+ 2018||this repo| |
|--|--|--|--|--|
||CIFAR-10 4000 labels|SVHN 1000 labels|CIFAR-10 4000 labels|SVHN 1000 labels|
|Supervised|20.26 ±0.38|12.83 ±0.47|19.85|11.03
|Pi-Model|16.37 ±0.63|7.19 ±0.27|14.84|7.87
|Mean Teacher|15.87 ±0.28|5.65 ±0.47|14.28|5.83
|VAT|13.13 ±0.39|5.35 ±0.19|12.15|6.38

NOTE: Our implementation is different from Oliver+ 2018 as follows:
1. we use not only purely unlabeled data, but also labeled data as unlabeled data. (following Sohn+ 2020)
2. our VAT implementation follows Miyato+, but Oliver+ use KLD with different directions as the loss function.
see [issue](https://github.com/brain-research/realistic-ssl-evaluation/issues/27).
3. parameter initialization of WRN-28. (following Sohn+ 2020)

If you would like to evaluate the model with the same conditions as Oliver+ 2018, please see [this repo](https://github.com/perrying/realistic-ssl-evaluation-pytorch).

||Sohn+ 2020||this repo| |
|--|--|--|--|--|
||CIFAR-10 250 labels|CIFAR-10 4000 labels|CIFAR-10 250 labels|CIFAR-10 4000 labels|
|UDA|8.82±1.08|4.88±0.18 | N/A | 6.32
|FixMatch|5.07±0.65|4.26±0.05| N/A | 6.84

reported error rates are the median of last 20 checkpoints

# Citation
```
@misc{suzuki2020consistency,
    author = {Teppei Suzuki},
    title = {Consistency Regularization for Semi-supervised Learning with PyTorch},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/perrying/pytorch-consistency-regularization}},
}
```

# References
- Miyato, Takeru, et al. "Distributional smoothing with virtual adversarial training." arXiv preprint arXiv:1507.00677 (2015).
- Laine, Samuli, and Timo Aila. "Temporal ensembling for semi-supervised learning." arXiv preprint arXiv:1610.02242 (2016).
- Sajjadi, Mehdi, Mehran Javanmardi, and Tolga Tasdizen. "Regularization with stochastic transformations and perturbations for deep semi-supervised learning." Advances in neural information processing systems. 2016.
- Tarvainen, Antti, and Harri Valpola. "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." Advances in neural information processing systems. 2017.
- Miyato, Takeru, et al. "Virtual adversarial training: a regularization method for supervised and semi-supervised learning." IEEE transactions on pattern analysis and machine intelligence 41.8 (2018): 1979-1993.
- Oliver, Avital, et al. "Realistic evaluation of deep semi-supervised learning algorithms." Advances in Neural Information Processing Systems. 2018.
- Verma, Vikas, et al. "Interpolation consistency training for semi-supervised learning." arXiv preprint arXiv:1903.03825 (2019).
- Sohn, Kihyuk, et al. "Fixmatch: Simplifying semi-supervised learning with consistency and confidence." arXiv preprint arXiv:2001.07685 (2020).
