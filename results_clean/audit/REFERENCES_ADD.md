# References to add

Both arXiv abstract pages were fetched on 2026-11-19 to confirm title, authors (in order) and year. Neither page lists
a journal reference or comments field, so both entries are `@misc` arXiv preprints. The key and field style follow
`report/overleaf/demo.bib`. Neither paper is in that file yet; `demo.bib` itself was not edited.

| arXiv id | title (as on the page) | authors (in order) | v1 date | latest version |
|---|---|---|---|---|
| 1803.01271 | An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling | Shaojie Bai, J. Zico Kolter, Vladlen Koltun | 2018-03-04 | v2, 2018-04-19 |
| 1609.03499 | WaveNet: A Generative Model for Raw Audio | Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu | 2016-09-12 | v2, 2016-09-19 |

```bibtex
@misc{bai2018,
  author        = {Bai, Shaojie and Kolter, J. Zico and Koltun, Vladlen},
  title         = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
  howpublished  = {arXiv preprint arXiv:1803.01271},
  year          = {2018},
  eprint        = {1803.01271},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/1803.01271}
}

@misc{vandenoord2016,
  author        = {van den Oord, Aaron and Dieleman, Sander and Zen, Heiga and Simonyan, Karen and Vinyals, Oriol and Graves, Alex and Kalchbrenner, Nal and Senior, Andrew and Kavukcuoglu, Koray},
  title         = {{WaveNet}: A Generative Model for Raw Audio},
  howpublished  = {arXiv preprint arXiv:1609.03499},
  year          = {2016},
  eprint        = {1609.03499},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/1609.03499}
}
```

Where they would be cited: the TCN backbone (`src/models/backbone.py:TCNForecaster`: causal dilated convolutions with
residual blocks) follows Bai et al. 2018. The dilated causal convolution comes from WaveNet (van den Oord et al. 2016).
