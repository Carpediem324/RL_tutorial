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
pip install "numpy<2"
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