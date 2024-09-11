
test if base inmage works

```bash
docker run --gpus all -it --rm pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel /bin/nvidia-smi && python -c "import torch; print(torch.cuda.is_available())"
```

make dockerfile with local repo

```bash
docker build -t mamba-vision .
```

run dummy test

```bash
docker run --gpus all -it --rm mamba-vision python mambavision/dummy_test.py
```

download imagenette https://github.com/fastai/imagenette

```bash
cd ~/dev/data/
mkdir -p imagenette
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz -O imagenette/imagenette2.tgz
tar -xzf imagenette/imagenette2.tgz -C imagenette
```

run training script

```bash
docker run --gpus all -it --rm -v ~/dev/data/imagenette2:/imagenette2 mamba-vision ./mambavision/train-imagenette2.sh
```

run training script with mounted log dir

```bash
docker run --gpus all -it --rm -v ~/dev/data/imagenette2:/imagenette2 -v ~/dev/data/mambavision_imagenette2:/mambavision_imagenette2 mamba-vision ./mambavision/train-imagenette2.sh
```

run tensorboard outside container using conda env

```bash
conda create --name tb python=3.10 tensorboard
conda activate tb
conda install numpy=1.24 # downgrade numpy to prevent error
tensorboard --logdir mambavision_imagenette2/testrun_my_experiment
```

https://paperswithcode.com/sota/image-classification-on-imagenette



