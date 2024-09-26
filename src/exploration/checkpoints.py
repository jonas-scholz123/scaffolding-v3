# %%
from mlbnb.checkpoint import CheckpointManager
from mlbnb.paths import ExperimentPath

path = ExperimentPath(
    "/home/jonas/Documents/code/scaffolding-v3/_output/DATA:data_provider=Era5DataProvider_val_fraction=0.1_train_range=2006-01-01_2011-01-01_test_range=2011-01-01_2012-01-01_num_times=10000_task_loader=TaskLoader_discrete_xarray_sampling=false_trainloader=DataLoader_batch_size=4_shuffle=true_num/workers=0_include_aux_at_targets=true_include_context_in_target=true_ppu=150_hires_ppu=2000_cache=false/MODEL:ConvNP_internal_density=150_unet_channels=64_64_64_64_aux_t_mlp_layers=64_64_64_likelihood=cnp_encoder_scales=0.0033333333333333335_decoder_scale=0.0033333333333333335/OPTIMIZER:Adam_lr=0.0001/SCHEDULER:StepLR_step_size=10_gamma=0.5/EXECUTION:device=cuda_dry_run=false_seed=42/checkpoints/best.pt"
)

checkpoint_manager = CheckpointManager(path)
