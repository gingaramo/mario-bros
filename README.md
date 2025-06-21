# Training Mario

![mario](./media/images/training_mario.png)

This repository contains a my own implementation of RL papers for Mario Bros environment.

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
