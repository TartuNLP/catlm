# yarnnlm


## How to use

### First train

### Scoring multiple sentences (no category)

```python
from rnnlm import score_sents_nocat, loadModels

model = loadModels(modelInFile, dictInFile)

print(score_tmp( [list("I go to bed"), list("I sleep"), list("I go to bed")], model))
```
