# Install requirements
# Usage: >> sudo ./setup.sh
sudo apt update -y
sudo apt upgrade -y
sudo apt install python3-dev -y
sudo apt install python3-pip -y
# Required for numpy
sudo apt install libatlas-base-dev -y
# Upgrade pip
python3 -m pip install -U pip
# Used setuptools version > 46.0
python3 -m pip --no-cache-dir install --ignore-install setuptools
# Used wheel version > 0.34
python3 -m pip --no-cache-dir install --ignore-install wheel
# Used joblib version > 0.14
python3 -m pip --no-cache-dir install --ignore-install joblib
# Used numpy version > 1.18
python3 -m pip --no-cache-dir install --ignore-install numpy
# Used pandas version > 0.25
python3 -m pip --no-cache-dir install --ignore-install pandas
# Had issues with scipy 1.4 and numpy, specifically 'no module named 'numpy.testing...''. BTW, scipy may attempt to reinstall numpy, but that's ok
python3 -m pip --no-cache-dir install --ignore-install 'scipy==1.3.3'
# Used cython version > 0.29
python3 -m pip --no-cache-dir install --ignore-install cython
# Keras 2.3 and TF 2.0 do not work with some libraries, like Mask R-CNN. BTW, do not use --ignore-installed or Keras will try to install scipy 1.4
python3 -m pip --no-cache-dir install 'keras<2.3'
#  Used version > 0.22. BTW, do not use --ignore-installed or scikit-learn will try to install scipy 1.4
python3 -m pip --no-cache-dir install scikit-learn
# Use wrapt version > 1.12
python3 -m pip --no-cache-dir install --ignore-install wrapt
# TF 2.0 and Keras 2.3 do not work with some libraries, like Mask R-CNN. BTW, do not use --ignore-installed or TF will try to install scipy 1.4
python3 -m pip --no-cache-dir install 'tensorflow<2.0'