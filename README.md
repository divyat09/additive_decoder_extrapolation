# Additive Decoders for Disentanglement and Extrapolation

## Reproduce Results for the ScalarLatents Dataset

### Generate Dataset
- python data/balls_dataset.py --latent_case supp_l_shape
- python data/balls_dataset.py --latent_case supp_extrapolate

### Train Models
- python scripts/scalar_disentanglement_exps.py

### Evaluate Models
- bash scalar_disentanglement_eval_launcher.sh

### Plotting Results
- python scripts/gen_draft_results.py --results_case violin_plot_2d

## Reproduce Results for the BlockLatents Dataset

### Generate Dataset
- python data/balls_dataset.py --latent_case latent_traversal
- python data/balls_dataset.py --latent_case supp_iid_no_occ
- python data/balls_dataset.py --latent_case supp_scm_linear

### Train Models
- python scripts/block_disentanglement_exps.py

### Evaluate Models
- bash block_disentanglement_eval_launcher.sh

### Plotting Results
- python scripts/gen_draft_results.py --results_case violin_plot_4d
