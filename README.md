# Counterfactual Explanations and Model Multiplicity: a Relational Verification View

Repository for the KR 2023 paper "Counterfactual Explanations and Model Multiplicity: a Relational Verification View".


## Training scripts 

Training and CFX generation using gradient-based algorithms are implemented in the main.py script.
To obtain all the parameters, run the following:

```
python main.py -h
```

This will return:

```
positional arguments:
  ds            Dataset name.
  dp            Path to dataset.
  mp            Path where model should be loaded/saved.
  lp            Path where logs should be loaded/saved.
  a             Algorithm used to generate counterfactuals.

optional arguments:
  -h, --help    show this help message and exit
  -train        Controls whether model is trained anew. Default:False.
  -nmods NMODS  Number of models to be considered. Default: 2.
  -nexps NEXPS  Number of cfx to be generated. Default: 1.
```

## Example

Say you want to train two models from scratch, and generate 1 explanation using the wachter method. Type:

```
python main.py ../datasets/german/ ../models/german/ ../results/german.txt wachter -nexp 1 -train
```

Models will be trained and saved in "../models/german/". Then, explanations for these models will be generated using the wachter method. A summary of results is printed in "../results/german.txt".


## Authors

[Francesco Leofante](https://fraleo.github.io),
[Elena Botoeva](https://www.kent.ac.uk/computing/people/3838/botoeva-elena),
[Vineet Rajani](https://vineetrajani.github.io/)

Do not hesitate to contact us if you have problems using this code, or if you find bugs :)


## Citing OMTPlan

If you decide to use this code for your experiments, please cite

	@inproceedings{LeofanteBotoevaRajani23,
  author       = {Francesco Leofante and Elena Botoeva and Vineet Rajani},

  title        = {Counterfactual Explanations and Model Multiplicity: a Relational Verification View},
  booktitle    = {Proceedings of the 20th International Conference on Principles of
                  Knowledge Representation and Reasoning, {KR} 2023, Rhodes, Greece.
                  September 02 - 08, 2023},
  year         = {2023}
  }


