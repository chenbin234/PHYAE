"""
This file is to train the model with the Turbo AutoEncoder architecture
Paper: Turbo Autoencoder: Deep learning based channel codes for point-to-point communication channels
URL: http://arxiv.org/abs/1911.03038
"""

# 1. Import Libraries

import sys

sys.path.append("src/")
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from trainer import (
    HybridBCELogSumExpLoss,
    HybridBCEMaxLoss,
    HybridBCEPNormLoss,
    HybridBCEProductLoss,
    HybridBCESmoothMaxLoss,
    HybridBCESoftBLERLoss,
    LogSumExpLoss,
    MaxLoss,
    PNormLoss,
    ProductLoss,
    SmoothMaxLoss,
    Trainer,
    broadcast_seed,
    ddp_setup,
    setup_seed,
)

import wandb
from features.build_pytorch_dataset2 import InfobitDataset

# from models.SerialTurbo_AutoEncoder import Turbo_AutoEncoder, generate_permutation_array
from models.SerialTurbo_AutoEncoder_key2 import (
    Turbo_AutoEncoder,
    generate_permutation_array,
)

# from torchinfo import summary


# def print_model_summary(model, input_size):
#     print("{:<40} {:<25} {}".format("Layer (type)", "Output Shape", "Param #"))
#     print("=" * 80)
#     dummy_input = torch.zeros(*input_size)
#     hooks = []

#     def register_hook(module):
#         def hook(module, input, output):
#             class_name = module.__class__.__name__
#             output_shape = str(list(output.shape))
#             params = sum(p.numel() for p in module.parameters() if p.requires_grad)
#             if not isinstance(module, nn.Sequential) and not isinstance(
#                 module, nn.ModuleList
#             ):
#                 print("{:<40} {:<25} {:,}".format(class_name, output_shape, params))

#         if (
#             not isinstance(module, nn.Sequential)
#             and not isinstance(module, nn.ModuleList)
#             and not (module == model)
#         ):
#             hooks.append(module.register_forward_hook(hook))

#     model.apply(register_hook)
#     model(dummy_input)
#     for h in hooks:
#         h.remove()


def make(config):
    # test dataset
    test_dataset = InfobitDataset(num_samples=1e5, k=config.input_size)

    # create test dataloader
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Initialize the model
    model = Turbo_AutoEncoder(
        k=config.k,
        n=config.n,
        add_parity_check_bit=config.add_parity_check_bit,
        n_M=config.n_M,
        F=config.F,
        channel=config.channel,
        is_delay=config.is_delay,
        num_iteration=config.num_iteration,
        p_array_seed=config.p_array_seed,
        num_taps=config.num_taps,
        max_sampling_time_offset=config.max_sampling_time_offset,
    )
    print("p_array_seed:", config.p_array_seed)
    if config.add_parity_check_bit:
        print(
            "p_array:",
            generate_permutation_array(k=config.k + 1, seed=config.p_array_seed),
        )
    else:
        print(
            "p_array:", generate_permutation_array(k=config.k, seed=config.p_array_seed)
        )

    # print_model_summary(model, input_size=(1, 1, 64))
    # Print the model summary
    # dummy_x = torch.zeros(500, 64, 1)
    # summary(model, input_data=(dummy_x, 0, 20))

    if config.pretrained:
        model.load_state_dict(torch.load(config.pretrained)["model_state_dict"])
        print("Pretrained model loaded")

    # loss function selection
    if config.loss_function == "BCE":
        loss_fn = nn.BCELoss()
    elif config.loss_function == "MSE":
        loss_fn = nn.MSELoss()
    elif config.loss_function == "ProductLoss":
        loss_fn = ProductLoss()
    elif config.loss_function == "MaxLoss":
        loss_fn = MaxLoss()
    elif config.loss_function == "SmoothMaxLoss":
        loss_fn = SmoothMaxLoss(alpha=config.alpha)
    elif config.loss_function == "LogSumExpLoss":
        loss_fn = LogSumExpLoss()
    elif config.loss_function == "PNormLoss":
        loss_fn = PNormLoss(p=2)
    elif config.loss_function == "HybridBCESoftBLERLoss":
        loss_fn = HybridBCESoftBLERLoss(beta=config.beta)
    elif config.loss_function == "HybridBCEProductLoss":
        loss_fn = HybridBCEProductLoss(beta=config.beta)
    elif config.loss_function == "HybridBCEMaxLoss":
        loss_fn = HybridBCEMaxLoss(beta=config.beta)
    elif config.loss_function == "HybridBCESmoothMaxLoss":
        loss_fn = HybridBCESmoothMaxLoss(beta=config.beta)
    elif config.loss_function == "HybridBCELogSumExpLoss":
        loss_fn = HybridBCELogSumExpLoss(beta=config.beta)
    elif config.loss_function == "HybridBCEPNormLoss":
        loss_fn = HybridBCEPNormLoss(p=2, beta=config.beta)
    else:
        raise ValueError("Invalid loss function")

    # Make the loss and optimizer
    if config.channel == "AWGN":
        optimizer_encoder = optim.Adam(
            model.encoder.parameters(), lr=config.learning_rate
        )
        optimizer_decoder = optim.Adam(
            filter(lambda p: p.requires_grad, model.decoder.parameters()),
            lr=config.learning_rate,
        )
    elif config.channel == "proakis_c_channel" and config.is_delay:
        optimizer_encoder = optim.Adam(
            model.encoder.parameters(), lr=config.learning_rate
        )
        optimizer_decoder = optim.Adam(
            filter(lambda p: p.requires_grad, model.decoder.parameters()),
            lr=config.learning_rate,
        )
    else:
        raise ValueError("Invalid channel type")

    return (model, loss_fn, optimizer_encoder, optimizer_decoder, test_dataloader)


def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(
        project="Joint-detection-and-decoding-in-short-packet-communications",
        config=hyperparameters,
        group=hyperparameters["model_type"],
    ):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # os.environ["LOCAL_RANK"] = "0"
        # if config.use_ddp:
        #     ddp_setup()

        # make the model, data, and optimization problem
        (
            model,
            loss_fn,
            optimizer_encoder,
            optimizer_decoder,
            test_dataloader,
        ) = make(config)
        print(model)

        # and use them to train the model
        trainer = Trainer(
            config=config,
            model=model,
            loss_fn=loss_fn,
            optimizer_encoder=optimizer_encoder,
            optimizer_decoder=optimizer_decoder,
        )
        if config.channel == "AWGN":
            trainer.train_TurboAE()
        elif config.channel == "proakis_c_channel":
            trainer.train_TurboAE_RayleighBlockFading()
        else:
            raise ValueError("Invalid channel")

        if config.use_ddp:
            destroy_process_group()

        # inference on the test set
        trainer.inference(test_dataloader=test_dataloader)

        # finish logging to wandb
        wandb.finish()

    return model


if __name__ == "__main__":
    # setup_seed(42)

    # Initialize Weights and Biases
    wandb.login()

    time = datetime.now().strftime("%Y%m%d_%H_%M_%S")
    # defining model save location
    save_model_name = "Turbo_AutoEncoder_" + time

    config = dict(
        model_type="Turbo_AutoEncoder",
        description="Turbo Autoencoder, proakis_c_channel",
        trainable_parameters=0,
        epochs=100,
        num_samples=1000,
        batch_size=1000,
        learning_rate=1e-5,
        start_epoch=1,
        print_every=None,
        save_every=100,
        in_channels=1,
        input_size=64,
        k=64,  # number of information bits
        n=64,  # code length (complex)
        n_M=6,  #
        F=10,
        num_iteration=6,
        num_train_enc=100,  # Number of iterations for training the encoder in each epoch
        num_train_dec=500,  # Number of iterations for training the decoder in each epoch
        dec_num_unit=100,
        dec_num_layer=5,
        add_parity_check_bit=True,  # Whether to add a parity check bit to the input of the encoder
        train_enc_snr_high=15.0,  # dB
        train_enc_snr_low=15.0,  # dB
        train_dec_snr_high=15.0,  # dB
        train_dec_snr_low=10.0,  # dB
        test_snr_high=20,  # dB
        test_snr_low=0,  # dB
        channel="proakis_c_channel",  # Channel type: "AWGN" or "proakis_c_channel"
        max_sampling_time_offset=1.0,  # Maximum sampling time offset for pulse shaping (0.0, 0.49, or 1.0)
        num_taps=1,  # Number of taps for the Proakis channel
        is_delay=True,  # Whether to include delay in the channel
        alpha=0.01,
        beta=1,
        pretrained="./models/Turbo_AutoEncoder_20260325_13_28_38/Turbo_AutoEncoder_20260325_13_28_38_epoch700.pth",  # Path to the pretrained model： "./models/Turbo_AutoEncoder_20241205_10_38_41/Turbo_AutoEncoder_20241205_10_38_41_epoch10.pth"
        loss_function="HybridBCESmoothMaxLoss",  # "BCE", "MSE", "ProductLoss", "MaxLoss", "SmoothMaxLoss", "LogSumExpLoss", "PNormLoss", "HybridBCESoftBLERLoss", "HybridBCEProductLoss", "HybridBCEMaxLoss", "HybridBCESmoothMaxLoss", "HybridBCELogSumExpLoss", "HybridBCEPNormLoss"
        save_model_name=save_model_name,
        snapshot_path=f"./models/{save_model_name}" + "/" + save_model_name + "_epoch",
        use_ddp=False,
        workstation_GPU_index=0,
        p_array_seed=2,
        max_stuck_counter=35,
        seed=1,
    )
    # create the folder if it doesn't exist
    if config["use_ddp"]:
        ddp_setup()
        # agree on a base‐seed
        base_seed = broadcast_seed(config["seed"])
        setup_seed(base_seed)

        if torch.distributed.get_rank() == 0:
            os.makedirs(f"./models/{save_model_name}", exist_ok=True)
        torch.distributed.barrier()
    else:
        setup_seed(config["seed"])
        # create the folder if it doesn't exist
        if not os.path.exists(f"./models/{save_model_name}"):
            os.makedirs(f"./models/{save_model_name}", exist_ok=True)

    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config)
