# requiem2d
requiem2d is a python code created to fit HST/WFC3/G141 grism spectra along with all available HST, Spitzer and ALMA photometric measurements. It is capable of fitting spatially resolved stellar populations while including global and semi-resolved information within a fully Bayesian framework. The Bayesian model is discussed in our [paper](https://arxiv.org/abs/2008.02276).

requiem2d relies on [Grizli](https://github.com/gbrammer/grizli/) to analyze grism data. It also uses [pymc3](https://github.com/pymc-devs/pymc) to sample the posteriors. [FSPS](https://github.com/cconroy20/fsps) and [python-FSPS](https://github.com/dfm/python-fsps) are used for generating stellar population synthesis models. These are core dependencies of requiem2d, and they should be installed properly before requiem2d.
