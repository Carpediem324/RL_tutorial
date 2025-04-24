# Machine Learning Project
## Reinforced Learning

### Enviroment

|type|version|
|------|---|
|Python|3.10|
|tensorboard|2.19.0|
|pytorch|?|
|cuda|?|


### step 1. cartpole

```bash
pip3 install gymnasium[classic_control]
pip uninstall numpy
pip install numpy==1.22.4
```

cartpole_kr.py 
Learn
```
python3 1.cartpole/cartpole_kr.py
```
cartpole_evaluation.py
Evaluation
```
python3 1.cartpole/cartpole_kr.py
```

### step 2. Inverted pendulum with mujoco

```
pip3 install torchrl
pip3 install gym[mujoco]
pip3 install tqdm
pip3 install stable_baselines3
pip3 install mujoco_py
pip3 install pygame
```

https://www.roboti.us/license.html
mujoco200_linux설치 후 ~/.mujoco/로 복사
라이센스키도 받아서 복사


무조코2.1.0설치
https://github.com/google-deepmind/mujoco/releases?page=4

```
tar -xvzf mujoco210-linux-x86_64.tar.gz  -C ~/.mujoco/
```


```
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

```
sudo apt-get update && sudo apt-get install patchelf
```