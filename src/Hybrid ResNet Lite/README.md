## Hybrid Resnet Lite

#### Prepare the environment

`Python 3.6 +` 、`Pytorch 1.2 +`、`Python-pcl` 、`numpy`、`sklearn`、 `tqdm` ... packages are needed, you can prepare the environment with 

```bash
pip install -r requirements.txt
```
**Notice:**  To install the `python-pcl` package, we highly recommend that you install it from source at  https://github.com/strawlab/python-pcl

#### Prepare train and test data

You can access our public data at  `Baiduyun-Cloud` . The url is https://pan.baidu.com/s/1nYYVnSxpRYbd9iFPTO7kog  and keywords: `4fa0`

#### Usage

After you prepare the environment and the dataset, you can train your own model with:

```bash
python train.py --train_data_dir $your_train_data_dir$ --test_data_dir $your_test_data_dir$
```

You can test your trained model with:

```bash
python test_pre.py --train_data_dir $your_train_data_dir$ --test_data_dir $your_test_data_dir$ --pretrained_model $your_trained_model_path$
```



