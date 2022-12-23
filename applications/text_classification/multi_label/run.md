## 安装依赖包
pip install paddlepaddle==2.4.0
pip install scikit-learn==1.0.2
pip install  CommandNotFound
pip install data common dual tight prox wheel
pip install paddle paddlenlp==2.4
pip install cudatoolkit==10.2
pip install paddlepaddle-gpu
pip install -r requirements.txt


### doccano标注转训练数据
python doccano.py --doccano_file E:\zncsData\doccano\jsonl\onez-annualclassificationall.jsonl --save_dir ./data --splits 0.8 0.1 0.1 --task_type "multi_label"

python doccano.py --doccano_file E:\zncsData\doccano\jsonl\doccano.jsonl --save_dir  E:\zncsData\doccano\jsonl\data --splits 1 0 0 --task_type "hierarchical"
python doccano.py --doccano_file doccano.jsonl --save_dir ./data --splits 0.8 0.1 0.1 --task_type "multi_label"
python doccano.py --doccano_file doccano.jsonl --save_dir ./data --splits 0.8 0.1 0.1 --task_type "hierarchical"



- 卸载
python -m pip uninstall paddlepaddle

###  paddle 默认是cpu版本 需要安葬GPU版本
参考：https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/linux-pip.html#ruhechakannindehuanjing
#### 计算机环境查看
uname -m && cat /etc/*release
python -V
python -m ensurepip
python -m pip --version

python -m pip install paddlepaddle-gpu==2.4.0.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

#### 电脑GPU环境
##### 查看cuda
-  nvidia-smi  11.6
-  nvcc -V   10.1
-  cat /usr/local/cuda/version.txt   10.1
- 三个版本不一样
  
##### 查看安装cudnn
- 查看
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
  
- 安装
https://developer.nvidia.com/rdp/cudnn-archive
tar zxvf cudnn-10.1-linux-x64-v8.0.5.39.tgz -C 
  
sudo cp cuda/include/cudnn.h /usr/local/cuda-10.1/include
sudo cp cuda/include/* /usr/local/cuda-10.1/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.1/lib64
sudo chmod a+r /usr/local/cuda-10.1/include/cudnn.h 
sudo chmod a+r /usr/local/cuda-10.1/lib64/libcudnn*

#sudo cp cuda/include/cudnn.h /usr/local/cuda/include
#sudo cp cuda/include/* /usr/local/cuda/include
#sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
#sudo chmod a+r /usr/local/cuda/include/cudnn.h 
#sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

sudo dpkg -i libcudnn8_8.0.5.39-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn8-dev_8.0.5.39-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn8-samples_8.0.5.39-1+cuda10.1_amd64.deb


##### 安装paddlepaddle-gpu
- https://pypi.tuna.tsinghua.edu.cn/simple/paddlepaddle-gpu/
- 旧版 https://www.paddlepaddle.org.cn/install/old?docurl=/documentation/docs/zh/install/pip/windows-pip.html

- CUDA 10.1
python -m pip install paddlepaddle-gpu==2.3.2.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

##### 卸载paddlepaddle
python -m pip uninstall paddlepaddle

## 运行命令
#### 训练 train
- 默认(训练轮次 2)
python train.py  --dataset_dir "data"  --device "cpu" --max_seq_length 128   --model_name "ernie-3.0-medium-zh"  --batch_size 32   --early_stop   --epochs 1

python train.py  --dataset_dir "data"  --device "cpu" --max_seq_length 128   --model_name "ernie-3.0-medium-zh"  --batch_size 32   --early_stop   --early_stop_nums 2

- cpu
python -m paddle.distributed.launch --nproc_per_node 4 --backend "gloo" train.py  --dataset_dir "data"  --device "cpu" --max_seq_length 128   --model_name "ernie-3.0-medium-zh"  --batch_size 32 --early_stop  --epochs 1

- gpu(驱动有问题、还不行)
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" train.py --dataset_dir "data" --device "gpu" --max_seq_length 128 --model_name "ernie-3.0-medium-zh" --batch_size 32 --early_stop --epochs 100


#### 评估 evaluate
python analysis/evaluate.py --device "cpu" --max_seq_length 128 --batch_size 32 --bad_case_file "bad_case.txt" --dataset_dir "data" --params_path "./checkpoint"


#### 预测
python predict.py --device "cpu" --max_seq_length 128 --batch_size 32 --dataset_dir "data"