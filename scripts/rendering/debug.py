# Third Party Library
import numpy as np
import PIL.Image


def main() -> None:
    img = PIL.Image.open("out_depth0001.png")
    print(np.array(img).shape)

    unique = np.unique(np.array(img), return_counts=True)
    for (v, c) in zip(unique[0], unique[1]):
        print(f"{v}: {c}")


if __name__ == "__main__":
    main()
