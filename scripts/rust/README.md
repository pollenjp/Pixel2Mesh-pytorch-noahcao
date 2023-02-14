# rust scripts

## trim-left

```sh
cargo run \
  --bin trim-left \
  -- \
  --src-file "../../datasets/data/shapenet/meta/train_tf.txt" \
  --dst-file "./out/data.txt"
```

## small-dataset-extraction

```sh
cargo run \
  --bin small-dataset-extraction \
  -- \
  --meta-data-file "../../datasets/data/shapenet/meta/train_tf.txt" \
  --out-file "./out/out.txt"
```

## select-categories

```sh
cargo run \
  --bin select-categories \
  -- \
  --src-file "./out/data.txt" \
  --dst-file "./out/data_sub.txt" \
  --categories "02691156"
```

## split-dataset

```sh
cargo run \
  --bin split-dataset \
  -- \
  --src-file "./out/data_sub.txt" \
  --dst-train-file "./out/train.txt" \
  --dst-test-file "./out/val.txt" \
  --train-ratio 0.8
```
