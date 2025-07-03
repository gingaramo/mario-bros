# Training Mario

![mario](./media/images/training_mario.png)

This repository contains a my own implementation of RL papers for Mario Bros environment.

# How to install

Follow these steps to set up the project:

1. **Install [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html)**  
  Download and install Conda if you don't have it already.

2. **Clone the repository**  
  ```bash
  git clone https://github.com/gingaramo/mario-bros.git
  cd mario-bros
  ```

3. **Install dependencies**  
  Create and activate the environment using the provided `environment.yml`:
  ```bash
  conda env create -f environment.yml
  conda activate mario-bros
  ```

4. **Apply necessary patches**  
  Run the patch script to apply required patches:
  ```bash
  ./src/apply_patches.sh
  ```

# How to run

1) Modify or create your own agent in `agents/` folder.

2) Run

```bash
$> python main.py --config agents/basic.yml
```

3) Once completed you'll have two new files in `runs/` folder `profile_{date}.data` and `config_{date}.yml`.

# Profiling

```bash
sudo py-spy record -o profile.svg -- python main.py --config agents/basic.yaml
```
