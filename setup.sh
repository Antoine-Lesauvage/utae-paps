#pip install -U pip setuptools wheel
#pip install torchvision 
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
git clone https://github.com/rusty1s/pytorch_scatter.git
cd pytorch_scatter
python setup.py install
cd ..
pip install -r requirements.txt
pip install earthengine-api geemap numpy
pip install tqdm
pip install scikit-learn seaborn opencv-python-headless