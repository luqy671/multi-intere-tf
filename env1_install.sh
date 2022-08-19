conda create -n env1 python=3.6
conda activate env1
conda install -c pytorch faiss-gpu cudatoolkit=10.1
pip install --upgrade tensorflow-gpu==1.15