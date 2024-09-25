# Setting up AI Models on Keeling

Author: Megha Rao (megha4@illinois.edu), Yudi Mao (yudimao2@illinois.edu), Mingfei Ren (mingfei5@illinois.edu)

Instructions and files necessary for setting up Google's NeuralGCM and NVIDIA's Earth2Mip AI Model Forecasting tools on UIUC's Keeling supercomputer. This document could help you prepare to use GPU on Keeling. We providded two examples (earth2mip & NeuralGCM) to help you running the models. 

## Setting up
- Log into Keeling:
```
ssh -Y netID@keeling.earth.illinois.edu
```
- Make sure you have Miniconda3 downloaded

You can check by using ls in your home directory, you should see Miniconda3 show up in your list of files

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Miniconda is important for package management – a lot of these AI models use a bunch of different python libraries for the code to run, Miniconda will help you to create different environments that ensure all of your packages will work with each other AND with the version of Python needed for the AI model.

- Activate L40S GPU node:

```
qlogin -p l40s -N 1 -n 96 --gres=gpu:L40S:1 --mem=250000 --time=48:00:00
```
MAKE SURE YOU ARE IN THE L40S GPU NODE OTHERWISE EARTH2MIP WILL NOT WORK AND NEURALGCM WILL TAKE A VERY LONG TIME.

- Activate conda installation:

```
source .bashrc
```
- Find the hostname of the node – remember this! This is how you will connect with Jupyter Notebook later

```
hostname
```
- Clone our git repository – this will help you to set up your conda environments with the necessary packages and dependencies:

```
git clone https://github.com/megham1nd/ai_modeling_on_keeling.git
```
- Go to the resulting folder that was created on your machine

```
cd ai_modeling_on_keeling
```
- Create your conda environments using the .yml files in the folder – make sure you are using the correct file! Match your AI model AND your system type to the correct file

```
conda env create -n name_of_your_environment -f=your_environment_file.yml python=3.10
```

- Activate the environment you just created

```
conda activate name_of_your_environment
```

- Now you are ready to connect with Jupyter Notebook!

## Setting up Conda Environments

- Download Miniconda3

- Download the following packages for each AI model:


||Earth2Mip|NeuralGCM|
|:------:|:------:|:---------:|
|General packages:|jupyter <br>ipykernel <br>mratplotlib <br>cartopy <br>earth2mip* |<br>neuralgcm <br>dinosaur-dycore <br>gcsfs |
|NVIDIA/Google dependencies:|torch-harmonics<br>pytorch<br>nvidia-apex**|Jax (jax[cuda12/cuda11])<br>Jaxlib|
|Other:|django-environ <br>gcc_linux-64 <br>gxx_linux-64 <br>onnxruntime-gpu*** <br>ruamel.yaml|flax|

\* It is important that you should satisfy: python --version <= 3.10
\** The nvidia-apex is different from the apex package, which is installed by `pip install apex`. You can install it via Nvidia’s github webset:
```
git clone git@github.com:NVIDIA/apex.git
cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

\*** The onnxruntime-gpu package is necessary to access Pangu-weather model via earth2mip.


## Connect with Jupyter Notebook

- Host your Jupyter notebook – replace the XXXX with any 4-digit port number; it doesn’t matter what it is, just make sure to remember it
 ```
 jupyter notebook --port=XXXX --no-browser --ip=127.0.0.1
 ```
- Copy the link that the terminal gives you – this is what you will put in to connect with VSCode

- Open up a NEW terminal window

- Run the following, replacing the XXXX with the port number you used in Step 1, netID with your netID, and hostname with what you got from Step 5 of Setting Up:
```
ssh -L YYYY:127.0.0.1:XXXX netID@keeling.earth.illinois.edu ssh -L XXXX:127.0.0.1:XXXX hostname
```
```
ssh -L 2002:127.0.0.1:3141 megha4@keeling.earth.illinois.edu ssh -L 3141:127.0.0.1:3141 keeling-gpu08
```
YYYY is your local port number, you can set it whatever you like.

- Open up VSCode and connect using the URL you copied from Step 2.

- You are all set to run your notebook! Try out the test cases from our git repository first to make sure everything is working properly.

Another method for setting code environment for VSCode on Keeling:

If you want to use VS Code on Keeling, you can download [VS Code](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://code.visualstudio.com/&ved=2ahUKEwiUyZWKjKKIAxUx4MkDHc7yKgwQFnoECAgQAQ&usg=AOvVaw15O90sm1ios8AUpw56hCml) on its page first. Then search for SSH in the VS Code Extensions Marketplace. Find the Remote-SSH extension published by Microsoft and click 'Install' to install it. After installation, find the icon on the left side and click it. Click the `+` icon in the upper right corner, then enter `username@server_IP_address` and press Enter. You will be prompted to enter the password. Simply input the password corresponding to the username you just entered.


### Base Environment: CUDA & CuDNN Installation

 Lots of AI programs (for example, earth2mip) depend on CUDA & CuDNN to accelerate computing. CUDA is a platform and programming model for CUDA-enabled GPUs. The Nvidia CUDA Deep Neural Network library (cuDNN) ia a GPU-accelerated library of primitives for deep neural networks.

 Because the latest version of CUDA which support CentOS 7 (Keeling’s Operation System) is 12.4, you should download it via Nvidia’s web. One available CuDNN version is [cudnn-linux-x86_64-8.9.7.29_cuda12](https://developer.nvidia.com/cuda-12-4-0-download-archive). 

 Another key issue is that we do not have sudo permission on Keeling, so you need to be careful on the path of the CUDA installation. We recommend you to install CUDA follow the instructions below:

 ```
# Create a local folder for installation
mkdir -p ~/local/cuda
mkdir -p ~/tmp/cuda-install

# Download cuda
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Install cuda
sh cuda_12.4.0_550.54.14_linux.run --silent --toolkit --toolkitpath=$HOME/local/cuda
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
 ```

Then, add the following to `.bashrc` to :

```
export PATH=$HOME/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/cuda/lib64:$LD_LIBRARY_PATH
```

IF YOU RUN INTO ERRORS:
```
conda uninstall torchvision
pip install torchvision==0.19.1
conda install -c anaconda cudnn
```
Your jupyter notebook should recognize the cuda toolkit; if you are unsure whether or not this is the case, add the following to a cell at the top of your code:
```
import torch
print(torch.cuda.is_available())
```
This should print `True`

After downloading the CuDNN file, you can extract it and install it by following the instructions:

```
tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
```
```
# Copy the header files
cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/cudnn*.h $HOME/local/cuda/include/
# Copy the library files
cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/libcudnn*.so* $HOME/local/cuda/lib64/
```
```
# Add operate permission and check the installation
chmod a+r $HOME/local/cuda/include/cudnn*.h
chmod a+rx $HOME/local/cuda/lib64/libcudnn*.so*
ls -l $HOME/local/cuda/lib64/libcudnn*
```
Now you have finished the installation!
