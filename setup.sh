# Install requirements
# Usage: >> sudo ./setup.sh
sudo apt update -y
sudo apt upgrade -y
sudo apt install python3-dev -y
sudo apt install python3-pip -y
sudo apt install libatlas-base-dev -y # Required for numpy
sudo pip install -U pip # Upgrade pip
sudo pip3 install -U pip # Upgrade pip3
sudo pip --no-cache-dir install --ignore-install setuptools # Use version > 46.0
sudo pip --no-cache-dir install --ignore-install wheel # Use version > 0.34
sudo pip --no-cache-dir install --ignore-install joblib # Use version > 0.14
sudo pip --no-cache-dir install --ignore-install numpy # Use version > 1.18
sudo pip --no-cache-dir install --ignore-install pandas # Use version > 0.25
sudo pip --no-cache-dir install --ignore-install 'scipy==1.3.3' # Had issues with scipy 1.4 and numpy, specifically 'no module named 'numpy.testing...''. BTW, scipy may attempt to reinstall numpy, but that's ok
sudo pip --no-cache-dir install --ignore-install cython # Use version > 0.29
sudo pip --no-cache-dir install 'keras<2.3' # Keras 2.3 and TF 2.0 do not work with some libraries, like Mask R-CNN. BTW, do not use --ignore-installed or Keras will try to install scipy 1.4
sudo pip --no-cache-dir install scikit-learn #  Used version > 0.22. BTW, do not use --ignore-installed or scikit-learn will try to install scipy 1.4
sudo pip --no-cache-dir install --ignore-install wrapt # Use version > 1.12
sudo pip --no-cache-dir install 'tensorflow<2.0' # TF 2.0 and Keras 2.3 do not work with some libraries, like Mask R-CNN. BTW, do not use --ignore-installed or TF will try to install scipy 1.4