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

**3. Train (configuration can be changed in `config.py` - batch size, learning rate, epochs, etc.):**

```
python train.py
```

**4. Test:**
   
```
python test.py
```



## Paper:

[[1](https://arxiv.org/abs/1706.03762)] 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
Attention Is All You Need. 
_arXiv:1706.03762 [cs.CL]_

