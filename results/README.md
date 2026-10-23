# Assignment experiment results

These files are outputs from the Task 5 and optional Task 6 scripts in this project. The MNIST training and test data are the local IDX files under `data/`.

## Task 5: generated digits

The comparison figures show 15 generated examples per digit class, sampled from models trained on the full MNIST training set with `alpha=1`, `alpha=2`, and `alpha=10`. The alpha=2 model also produced the mode and expected-pixel plots. Each model was trained once for this run; samples were generated with the `nb_generate` implementation in `a01_functions.py`.

| File | What it shows |
| --- | --- |
| `generated_alpha_1.png` | Samples generated with no additive smoothing (`alpha=1`). |
| `generated_alpha_2.png` | Samples generated with add-one smoothing (`alpha=2`); this is the assignment's default example. |
| `generated_alpha_10.png` | Samples generated with stronger smoothing (`alpha=10`). |
| `most_likely_alpha_2.png` | For each class and pixel, the most probable pixel value under the alpha=2 model. |
| `expected_value_alpha_2.png` | The expected pixel intensity for each class under the alpha=2 model. |

The generated examples use NumPy's random sampling without a fixed seed, so rerunning the script can produce different samples. The actual plots show recognizable class structure at alpha=1 and alpha=2; the alpha=10 examples appear noisier and less class-specific. The mode and expected-value plots show recognizable digit prototypes.

Task 5a (implementing the generator) and Task 5b (generating and plotting examples, then comparing and discussing smoothing values) are graded assignment requirements. These plots support Task 5b; the code and written discussion remain in the assignment sources/submission template.

## Task 6: optional sampled completions

| File | What it shows |
| --- | --- |
| `task6_optional_completions.png` | The first MNIST test image, its observed first 14 rows with the remaining pixels hidden, and five sampled completions from the Naive Bayes conditional distribution. The run used `alpha=2` and NumPy random seed 0. The image's true label and the most probable class from the visible prefix were both 7. |

The Task 6 formulas for full-feature classification, prefix-only classification, and the missing-suffix distribution are graded requirements and are written in `a01_6.py`. Sampling example images and discussing the completions is explicitly optional in the assignment; this figure documents that optional experiment.

## Reproduce

From the project root, run the `a01_5.py` cells to regenerate the Task 5 figures and the `a01_6.py` optional sampling cell to regenerate the Task 6 completion figure. Both scripts use the local helper and MNIST data. They save the images under `task5_plots/` and `task6_optional_completions.png`, respectively; copy the updated outputs into this directory if you want to refresh the checked-in results.
