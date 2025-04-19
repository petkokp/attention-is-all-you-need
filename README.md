# "Attention is all you need" implementation

## Rasul Alakbarli, Mahammad Nuriyev, Petko Petkov

**1. Install libraries:**

```
pip install -r requirements.txt
```

**2. Download spacy language pipelines:**

```
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

**3. Train:**

All hyperparameters (batch size, learning rate, epochs, optimizer settings, etc.) can be configured via command-line flags. Use `-h` or `--help` to see organized groups:

```bash
python train.py --help
```

To train with default settings:

```bash
python train.py
```

**4. Test:**

```bash
python test.py --exp_path experiments/<your_experiment>
```

**4. Plot results (plots are added to the experiment's directory in `experiments`):**

```bash
python plot.py --exp_path experiments/<your_experiment>
```

## Paper:

[[1](https://arxiv.org/abs/1706.03762)]
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
Attention Is All You Need.
_arXiv:1706.03762 [cs.CL]_
