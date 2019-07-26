# tf-eager-mnist

See https://www.tensorflow.org/guide/eager#train_a_model

## Run in Pipenv (CPU)

Requirements:

* python&emsp;&emsp;3.6.x
* pipenv&emsp;&emsp;2018.11.26

```bash
$ git clone https://github.com/kencyke/tf-eager-mnist.git
$ env PIPENV_VENV_IN_PROJECT=true pipenv install -d --python 3.6
$ mkdir ./output
$ pipenv run python tf_eager_mnist.py --output ./output
```

## Run in Docker (GPU)

Requirements: 

* See https://github.com/NVIDIA/nvidia-docker#quickstart

```bash
$ git clone https://github.com/kencyke/tf-eager-mnist.git
$ mkdir ./output
$ docker build -t xxx/yyy:tag -f ./dockerfile/gpu.Dockerfile .
$ docker run -it --rm --gpus all -v "$PWD":/tmp/test -w /tmp/test xxx/yyy:tag python tf_eager_mnist.py --output ./output
```