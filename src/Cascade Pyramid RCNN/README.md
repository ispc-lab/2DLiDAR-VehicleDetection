### Requirements:

- PyTorch 1.0 
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9

```bash
pip install ninja yacs cython matplotlib

conda install pytorch-1.0.0 -c pytorch

cd ~/github
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install

cd ~/github
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

cd maskrcnn-benchmark
python setup.py build develop
```

### Multi-GPU training

```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_maskrcnn_benchmark/tools/train_net.py --config-file "path/to/config/file.yaml"
```


