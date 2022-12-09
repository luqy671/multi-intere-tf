conda create -n env1 python=3.6
conda activate env1
conda install -c pytorch faiss-gpu cudatoolkit=10.1
pip install --upgrade tensorflow-gpu==1.15

pip install tensorflow-gpu==1.15 --default-timeout=100 -i https://pypi.tuna.tsinghua.edu.cn/simple