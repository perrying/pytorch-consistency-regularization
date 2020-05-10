# Consistency Regularizations for Semi-supervised Learning with PyTorch
This repositrory includes consistency regularization algorithms for semi-supervised learning:
- Pi-Model
- Pseudo-label
- Mean Teacher
- Virtual Adversarial Training
- Interpolation Consistency Training
- Unsupervised Data Augmentation
- FixMatch (with RandAugment)

Training and evaluation follow "Oliver et al., 2018" and FixMatch.

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
and the scripts in ```scripst/realistic-evaluation-setup``` are for training and evaluating a model with the "Oliver et al., 2018" setting.

If yor would like to train a model with own setting, please see ```parser.py```.

NOTE: ```train_test.py``` evaluates a model performance as median of last [1, 10, 20, 50] checkpoint accuracies (FixMatch setting),
and ```train_val_test.py``` evaluates the test accuracy of the best model on validation data (Oliver et al. 2018 setting).

# Performance
WIP

# Citation
```
@misc{ssl-consistency-regularization,
    author = {Teppei Suzuki},
    title = {Consistency Regularizations for Semi-supervised Learning with PyTorch},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/perrying/ssl-consistency-regularization}},
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
