import argparse
from PIL import Image
from face_swap.api import FaceSwapper


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("model_dir")
    opts = cli.parse_args()

    faceswapper = FaceSwapper(opts.model_dir)

    while True:
        try:
            path_src = input("source: ")
            path_tgt = input("target: ")
            path_out = input("output: ")
        except KeyboardInterrupt:
            yes_exit = input('type "exit" to confirm exit: ')
            if yes_exit == "exit":
                break
            else:
                continue
        image_result = faceswapper(path_src, path_tgt)
        Image.fromarray(image_result).save(path_out)


if __name__ == "__main__":
    main()
