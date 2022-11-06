# Pixel2Mesh

forked from <https://github.com/noahcao/Pixel2Mesh>

## requirements

- CUDA 9.0

```sh
git clone --recurse-submodules git@github.com:pollenjp/pixel2mesh-pytorch-noahcao.git
make setup
```

## Run

```sh
poetry run python src/entrypoint_train.py \
    --name=check-resnet \
    --options=./experiments/default/resnet.yml
```

## Scripts

- rust
