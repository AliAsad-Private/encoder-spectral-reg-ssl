# Reproducing the reported runs

The YAML files in `configs/` hold default settings. The reported runs set placement,
regularization weight, seed and (for STL-10) batch size through `--overrides`.
The commands below reproduce each configuration in the paper (Appendix A, Table 6).

## SimCLR, CIFAR-100

    # Baseline (seeds 42, 43, 44)
    python scripts/train.py --config configs/simclr_cifar100.yaml --overrides seed=42

    # Encoder placement, regularization sweep (seed 42; lambda in 0.1, 0.25, 0.5, 1.0)
    python scripts/train.py --config configs/simclr_cifar100_dimreg.yaml \
        --overrides dim_reg_on=encoder dim_reg_weight=0.5 seed=42

    # Encoder placement, lambda = 0.5, additional seeds 43 and 44
    python scripts/train.py --config configs/simclr_cifar100_dimreg.yaml \
        --overrides dim_reg_on=encoder dim_reg_weight=0.5 seed=43

    # Projector placement control (lambda = 0.1, as shipped)
    python scripts/train.py --config configs/simclr_cifar100_dimreg.yaml

## VICReg and Barlow Twins, CIFAR-100

    python scripts/train.py --config configs/vicreg_cifar100.yaml
    python scripts/train.py --config configs/barlow_cifar100.yaml
    python scripts/train.py --config configs/barlow_cifar100_dimreg.yaml

## SimCLR, STL-10 (batch size 512)

    python scripts/train.py --config configs/simclr_stl10.yaml --overrides batch_size=512
    python scripts/train.py --config configs/simclr_stl10_dimreg.yaml --overrides batch_size=512

## Corruption robustness (CIFAR-100-C)

    python scripts/eval_robustness.py --checkpoint <path/to/epoch_399.pth> --output results.json

Figures: `scripts/make_paper_figures.py` (paths set via the RES, LOGS and CKPT environment variables).
