#!/bin/bash

# "py3tfgnt" - TF 2.0 nightly-preview w/ TF-Agents nightly

ARCH=`uname`
echo `date`

PYMAJOR=3
PYMINOR=6
TFMAJOR=2
TFMINOR=0
TFPATCH=0
CONDAENVNAME="py3tfgnt"
PYVER="${PYMAJOR}.${PYMINOR}"
CONDAINSTALL=""
MINIDIR=""

if [[ $ARCH == "Darwin" ]]; then
  CONDAINSTALL=Miniconda${PYMAJOR}-latest-MacOSX-x86_64.sh
  MINIDIR="miniconda${PYMAJOR}"
else
  echo "Only configured for Mac OSX so far."
  exit 1
fi

if [[ ! -d "$HOME/$MINIDIR" ]]; then
  wget https://repo.continuum.io/miniconda/$CONDAINSTALL
  bash $CONDAINSTALL -b -p $HOME/$MINIDIR
fi

export PATH="$HOME/$MINIDIR/bin:$PATH"

conda info --envs
conda remove --yes --name $CONDAENVNAME --all
conda create -q -y -n $CONDAENVNAME python=$PYVER
. activate $CONDAENVNAME

pip install --upgrade pip
pip install --no-cache-dir 'python-language-server[all]'

conda install -q -y ipython
conda install -q -y jupyter
conda install -q -y tqdm
conda install -q -y coverage
#conda install -q -y nbconvert==5.4.1  # may be needed for tornado issue

pip install --no-cache-dir 'gym[atari,box2d,classic_control]'
pip install --no-cache-dir --upgrade tf-nightly-2.0-preview
pip install --no-cache-dir --upgrade tf-agents-nightly
pip install --no-cache-dir --upgrade tfp-nightly
pip install --no-cache-dir --upgrade tensorflow-datasets
pip install --no-cache-dir mlflow
pip install --no-cache-dir matplotlib
pip install --no-cache-dir seaborn
pip install --no-cache-dir scikit-image

echo "Be sure to adjust your PATH and include $HOME/$MINIDIR/bin"
echo " e.g., export PATH=$HOME/$MINIDIR/bin:\$PATH"
echo "Activate the $CONDAENVNAME environment with"
echo "\$ source activate $CONDAENVNAME"
echo "and deactivate with"
echo "\$ source deactivate"

echo -e "\a"
