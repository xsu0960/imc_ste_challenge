# Key Results

## Mandatory Classification

| Task | Clean | Direct noisy | Best STE | Recovery | Retained clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 + ResNet18 | 95.62% | 61.42% | 90.63% | +29.21 pp | 94.78% |
| CIFAR-10 + EfficientNet-B0 | 91.13% | 38.93% | 86.12% | +47.19 pp | 94.50% |
| CIFAR-100 + ResNet18 | 78.32% | 62.98% | 69.96% | +6.98 pp | 89.33% |
| CIFAR-100 + EfficientNet-B0 | 71.14% | 27.76% | 61.60% | +33.84 pp | 86.59% |
| TinyImageNet + ResNet18 ImageNet-224 | 71.60% | 53.83% | 58.07% | +4.24 pp | 81.10% |
| TinyImageNet + EfficientNet-B0 ImageNet-224 | 81.00% | 0.50% | 76.30% | +75.80 pp | 94.20% |

## Optional Full Validation

| Task | Metric | Clean | Direct noisy | 1-epoch STE | Recovery |
| --- | --- | ---: | ---: | ---: | ---: |
| VOC2007 + Faster R-CNN | mAP50 | 72.59% | 28.72% | 37.53% | +8.81 pp |
| VOC2012 + DeepLabV3 | mIoU | 71.31% | 4.03% | 6.46% | +2.43 pp |

## Optional Paired Extension

| Task | Control | Candidate | Delta | CI95 | p-value | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| VOC2012 + DeepLabV3 online extension | 6.50% | 6.74% | +0.245 pp | +/- 0.099 pp | 0.0086 | significant |

## Engineering Efficiency

| Path | Step time | Relative to clean | Throughput | Incremental peak memory |
| --- | ---: | ---: | ---: | ---: |
| clean | 4.87 ms | 1.00x | 1643.94 images/s | 35.50 MiB |
| ste | 25.71 ms | 5.28x | 311.12 images/s | 278.35 MiB |
| online_profile | 47.96 ms | 9.86x | 166.80 images/s | 296.45 MiB |
