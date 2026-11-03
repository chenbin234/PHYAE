import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from inference.inference_utils_TurboAE import inference_loop

# def ddp_setup():
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
#     init_process_group(backend="nccl")


# Add this function after the imports
def ddp_setup(force_gloo=True):
    """
    Set up distributed training with appropriate backend selection
    """
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # For single-process jobs, gloo is more reliable
    if force_gloo:
        backend = "gloo"
        print(f"Using {backend} backend for distributed training")
    else:
        # NCCL is better for multi-GPU but can have shared memory issues
        backend = "nccl"

    # Set required environment variables
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    try:
        torch.distributed.init_process_group(backend=backend)
        print(f"Distributed training initialized with {backend} backend")
    except Exception as e:
        print(f"Failed to initialize with {backend}: {e}")
        if backend == "nccl":
            print("Falling back to gloo backend")
            torch.distributed.init_process_group(backend="gloo")


class Trainer:
    def __init__(
        self,
        config: wandb.sdk.wandb_config.Config,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        **kwargs,
    ) -> None:
        self.local_rank = int(
            os.environ.get("LOCAL_RANK", config.workstation_GPU_index)
        )
        self.global_rank = int(os.environ.get("RANK", config.workstation_GPU_index))
        self.config = config
        self.model = model.to(self.local_rank)
        self.loss_fn = loss_fn
        # initiate input **kwargs to class attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        # load snapshot if exists
        if os.path.exists(self.config.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.config.snapshot_path)

        # warp model with DDP id needed
        if self.config.use_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(self.config.snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["model_state_dict"])
        self.epochs_run = snapshot["epoch"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _generate_data_epoch(self):
        X_train = torch.randint(
            low=0,
            high=2,
            size=(self.config.num_samples, self.config.input_size, 1),
            dtype=torch.float32,
        ).to(self.local_rank)
        # convert data to DataLoader
        if self.config.use_ddp:
            X_train = DataLoader(
                X_train,
                batch_size=self.config.batch_size,
                shuffle=False,
                generator=torch.Generator().manual_seed(self.config.seed),
                sampler=DistributedSampler(X_train),
            )
        else:
            X_train = DataLoader(
                X_train,
                batch_size=self.config.batch_size,
                generator=torch.Generator().manual_seed(self.config.seed),
                shuffle=True,
            )
        return X_train

    def _run_batch(self, X, mode, channel="AWGN"):
        # 2 different modes: encoder and decoder
        if mode == "encoder":
            self.optimizer_encoder.zero_grad()
            SNR_db_low = self.config.train_enc_snr_low
            SNR_db_high = self.config.train_enc_snr_high
        else:
            self.optimizer_decoder.zero_grad()
            SNR_db_low = self.config.train_dec_snr_low
            SNR_db_high = self.config.train_dec_snr_high
        # forward pass
        if channel == "AWGN":
            predictions = self.model.forward(X, SNR_db_low, SNR_db_high)
            predictions = torch.clamp(predictions, 0.0, 1.0)
            loss = self.loss_fn(predictions, X)
            sync_error = 0
        elif channel == "proakis_c_channel" and self.config.is_delay:
            # ! print the batch size used for training
            # print(f"Batch size: {X.shape[0]}")
            predictions = self.model.forward(X, SNR_db_low, SNR_db_high)
            predictions = torch.clamp(predictions, 0.0, 1.0)

            if self.config.add_parity_check_bit:
                # add parity check bit to X in order to calculate loss
                # because the prediction predict the length of k+1
                X = add_even_parity_bit(X)

            loss = self.loss_fn(predictions, X)
            sync_error = 0
        # elif channel == "proakis_c_channel" and self.config.is_delay:
        #     if mode == "encoder":
        #         (
        #             predictions,
        #             predicted_starting_point_probability,
        #             true_starting_point_onehot,
        #         ) = self.model.forward(
        #             X, SNR_db_low, SNR_db_high, is_training_encoder=True
        #         )
        #         predictions = torch.clamp(predictions, 0.0, 1.0)
        #     else:
        #         (
        #             predictions,
        #             predicted_starting_point_probability,
        #             true_starting_point_onehot,
        #         ) = self.model.forward(
        #             X, SNR_db_low, SNR_db_high, is_training_encoder=False
        #         )
        #         predictions = torch.clamp(predictions, 0.0, 1.0)

        #     loss = self.loss_fn(
        #         predictions,
        #         X,
        #         predicted_starting_point_probability,
        #         true_starting_point_onehot,
        #     )
        #     # log the error between predicted and true starting points
        #     sync_error = calculate_sync_error_rate(
        #         predicted_starting_point_probability,
        #         true_starting_point_onehot,
        #     )
        else:
            raise ValueError("Invalid channel type")

        # calculate loss
        loss.backward()
        if mode == "encoder":
            self.optimizer_encoder.step()
            # # ! print the learning rate of the encoder
            # for param_group in self.optimizer_encoder.param_groups:
            #     print(f"Encoder learning rate: {param_group['lr']}")
        else:
            self.optimizer_decoder.step()
            # # ! print the learning rate of the decoder
            # for param_group in self.optimizer_decoder.param_groups:
            #     print(f"Decoder learning rate: {param_group['lr']}")
        return loss.item(), sync_error

    def _run_epoch(self, mode):
        # generate data for each epoch
        X_train = self._generate_data_epoch()
        # set model to training mode
        self.model.train()

        # ! update the learning rate
        for param_group in self.optimizer_encoder.param_groups:
            param_group["lr"] = self.config.learning_rate
        for param_group in self.optimizer_decoder.param_groups:
            param_group["lr"] = self.config.learning_rate

        # measure time
        start_time = time.time()
        train_loss_batches = []
        sync_error_batches = []
        # iterate over the data
        for batch_index, X in enumerate(X_train):
            loss, sync_error = self._run_batch(
                X.to(self.local_rank), mode, channel=self.config.channel
            )
            train_loss_batches.append(loss)
            sync_error_batches.append(sync_error)
        train_loss_per_epoch = sum(train_loss_batches) / len(train_loss_batches)
        sync_error_per_epoch = sum(sync_error_batches) / len(sync_error_batches)
        end_time = time.time()
        # print(
        #     f"\tGPU: {self.global_rank} | "
        #     f"\tEpoch: {self.config.current_epoch} | "
        #     f"\tAverage Train loss ({mode}): {train_loss_per_epoch} | "
        #     f"\tAverage Sync Error (Delay: {self.config.is_delay}): {sync_error_per_epoch} | "
        #     f"\tTime taken: {end_time - start_time:.2f} sec\n"
        # )
        return train_loss_per_epoch, sync_error_per_epoch

    def _save_snapshot(self, epoch):
        # Saving model
        print(f"Saving model [epoch {epoch}]")
        if self.config.use_ddp:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.module.state_dict(),
                },
                f"./models/{self.config.save_model_name}/{self.config.save_model_name}_epoch{epoch}.pth",
            )
        else:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                },
                f"./models/{self.config.save_model_name}/{self.config.save_model_name}_epoch{epoch}.pth",
            )

    def val_TurboAE(self):
        test_bce_loss = 0
        test_BER = 0
        # generate data for each epoch
        X_val = self._generate_data_epoch()
        # set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for batch_index, X in enumerate(X_val):
                # valdate on train_enc_snr_low
                SNR_db_low = self.config.train_enc_snr_low
                SNR_db_high = self.config.train_enc_snr_low
                predictions = self.model.forward(X, SNR_db_low, SNR_db_high)
                predictions = torch.clamp(predictions, 0.0, 1.0)
                test_bce_loss += self.loss_fn(predictions, X).item()
                # round the predictions to 0 or 1
                predictions_round = torch.round(predictions)
                test_BER += calculate_BER(predictions_round, X)
        test_bce_loss_avg = test_bce_loss / (
            int(self.config.num_samples / self.config.batch_size)
        )
        test_BER_avg = test_BER / (
            int(self.config.num_samples / self.config.batch_size)
        )
        return test_bce_loss_avg, test_BER_avg

    def train_TurboAE(self):
        # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.model, self.loss_fn, log="all", log_freq=5000)
        # record the number of trainable parameters in the model
        wandb.config.update(
            {"trainable_parameters": count_parameters(self.model)},
            allow_val_change=True,
        )
        print(f"Number of total parameters:     {total_parameters(self.model)}")
        print(f"Number of trainable parameters: {count_parameters(self.model)}")
        print("Starting training")

        for epoch in range(self.config.start_epoch, self.config.epochs + 1):
            if epoch > 1:
                wandb.config.update({"current_epoch": epoch}, allow_val_change=True)
            else:
                self.config.current_epoch = epoch

            # train for one epoch
            for i in range(self.config.num_train_enc):
                _, _ = self._run_epoch(mode="encoder")
            for i in range(self.config.num_train_dec):
                train_loss_per_epoch, _ = self._run_epoch(mode="decoder")

            # compute the validation loss, test on train_enc_snr_low
            test_bce_loss_avg, test_BER_avg = self.val_TurboAE()

            print(
                f"Epoch {epoch}/{self.config.epochs}: "
                f"Val. loss: {test_bce_loss_avg} "
                f"Val. BER ({self.config.train_enc_snr_low} dB): {test_BER_avg}"
            )

            # Log the train_loss_one_epoch and val_loss_one_epoch to wandb
            wandb.log(
                {
                    "train_loss": train_loss_per_epoch,
                    "val_loss": test_bce_loss_avg,
                    "val_BER": test_BER_avg,
                },
                step=epoch,
            )

            # save snapshot
            if self.local_rank == 0 and epoch % self.config.save_every == 0:
                self._save_snapshot(epoch)

    def val_TurboAE_RayleighBlockFading(self):
        test_bce_loss = 0
        test_BER = 0
        test_BLER = 0
        test_sync_error = 0
        # generate data for each epoch
        X_val = self._generate_data_epoch()
        # set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for batch_index, X in enumerate(X_val):
                # valdate on train_enc_snr_low
                SNR_db_low = self.config.train_enc_snr_low
                SNR_db_high = self.config.train_enc_snr_low
                if self.config.is_delay:
                    predictions = self.model.forward(X, SNR_db_low, SNR_db_high)
                    predictions = torch.clamp(predictions, 0.0, 1.0)
                    if self.config.add_parity_check_bit:
                        # add parity check bit to X in order to calculate loss
                        # because the prediction predict the length of k+1
                        X = add_even_parity_bit(X)
                    test_bce_loss += self.loss_fn(
                        predictions,
                        X,
                    ).item()
                    test_sync_error += 0
                elif not self.config.is_delay:
                    predictions = self.model.forward(X, SNR_db_low, SNR_db_high)
                    predictions = torch.clamp(predictions, 0.0, 1.0)
                    test_bce_loss += self.loss_fn(predictions, X).item()
                    test_sync_error = 0
                # round the predictions to 0 or 1
                predictions_round = torch.round(predictions)
                test_BER += calculate_BER(predictions_round, X)
                test_BLER += calculate_BLER(predictions_round, X)

        test_bce_loss_avg = test_bce_loss / (
            int(self.config.num_samples / self.config.batch_size)
        )
        test_BER_avg = test_BER / (
            int(self.config.num_samples / self.config.batch_size)
        )
        test_BLER_avg = test_BLER / (
            int(self.config.num_samples / self.config.batch_size)
        )
        test_sync_error_avg = test_sync_error / (
            int(self.config.num_samples / self.config.batch_size)
        )
        return test_bce_loss_avg, test_BER_avg, test_BLER_avg, test_sync_error_avg

    def train_TurboAE_RayleighBlockFading(self):
        # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.model, self.loss_fn, log="all", log_freq=5000)
        # record the number of trainable parameters in the model
        wandb.config.update(
            {"trainable_parameters": count_parameters(self.model)},
            allow_val_change=True,
        )
        print(f"Number of total parameters:     {total_parameters(self.model)}")
        print(f"Number of trainable parameters: {count_parameters(self.model)}")
        print("Starting training")
        stuck_counter = 0
        lowest_cost = 2  # set to a high value

        for epoch in range(self.config.start_epoch, self.config.epochs + 1):
            if epoch > 1:
                wandb.config.update({"current_epoch": epoch}, allow_val_change=True)
            else:
                self.config.current_epoch = epoch

            # train for one epoch
            for i in range(self.config.num_train_enc):
                _, _ = self._run_epoch(mode="encoder")
            for i in range(self.config.num_train_dec):
                train_loss_per_epoch, train_sync_error_per_epoch = self._run_epoch(
                    mode="decoder"
                )

            # compute the validation loss, test on train_enc_snr_low
            test_bce_loss_avg, test_BER_avg, test_BLER_avg, val_sync_error_per_epoch = (
                self.val_TurboAE_RayleighBlockFading()
            )

            # update batch_size, learning_rate, stuck_counter and other parameters if needed
            if epoch == 1:
                lowest_cost = test_bce_loss_avg
            if test_bce_loss_avg > lowest_cost:
                stuck_counter += 1
                if (
                    stuck_counter > self.config.max_stuck_counter
                    and self.config.batch_size < 990
                ):
                    print(
                        f"\nLoss Stuck for {stuck_counter} epochs, increase batch size\n"
                    )
                    wandb.config.update(
                        {"batch_size": self.config.batch_size * 2},
                        allow_val_change=True,
                    )
                    # self.config.batch_size = self.config.batch_size * 2
                    wandb.config.update(
                        {"num_samples": self.config.num_samples * 2},
                        allow_val_change=True,
                    )
                    # self.config.num_samples = self.config.num_samples * 2

                    stuck_counter = 0
                if (
                    stuck_counter > self.config.max_stuck_counter
                    and self.config.batch_size > 990
                    and self.config.learning_rate >= 1e-5
                ):
                    print(
                        f"\nLoss Stuck for {stuck_counter} epochs, reduce learning rate\n"
                    )
                    wandb.config.update(
                        {"learning_rate": self.config.learning_rate / 10},
                        allow_val_change=True,
                    )
                    # self.config.learning_rate = self.config.learning_rate / 10

                    stuck_counter = 0
                if self.config.learning_rate <= 1e-6:
                    print(
                        f"\nLoss Stuck for {stuck_counter} epochs, but stop updating batch size and learning rate\n"
                    )
            else:
                stuck_counter = 0
                lowest_cost = test_bce_loss_avg

            # print(
            #     f"Epoch: {epoch}/{self.config.epochs} "
            #     f"Val. loss: {test_bce_loss_avg} "
            #     f"Val. BER ({self.config.train_enc_snr_low} dB): {test_BER_avg} "
            #     f"Val. BLER ({self.config.train_enc_snr_low} dB): {test_BLER_avg} "
            #     f"Batch size: {self.config.batch_size} "
            #     f"Learning rate: {self.config.learning_rate} "
            #     f"Stuck: {stuck_counter} "
            # )

            print(
                f"Epoch: {epoch:>2}/{self.config.epochs:<2} | "
                f"Val. loss: {test_bce_loss_avg:>10.6f} | "
                f"Val. BER ({self.config.train_enc_snr_low:>2} dB): {test_BER_avg:>10.6f} | "
                f"Val. BLER ({self.config.train_enc_snr_low:>2} dB): {test_BLER_avg:>10.6f} | "
                f"Batch size: {self.config.batch_size:>4} | "
                f"Learning rate: {self.config.learning_rate:>10.6e} | "
                f"Stuck: {stuck_counter:>3}"
            )

            # Log the train_loss_one_epoch and val_loss_one_epoch to wandb
            wandb.log(
                {
                    "train_loss": train_loss_per_epoch,
                    "train_sync_error_rate": train_sync_error_per_epoch,
                    "val_loss": test_bce_loss_avg,
                    "val_BER": test_BER_avg,
                    "val_BLER": test_BLER_avg,
                    "Val_sync_error_rate": val_sync_error_per_epoch,
                },
                step=epoch,
            )

            # save snapshot
            if (
                self.local_rank == self.config.workstation_GPU_index
                and epoch % self.config.save_every == 0
            ):
                self._save_snapshot(epoch)

    def inference(self, test_dataloader):
        inference_loop(
            config=self.config,
            model=self.model,
            test_dataloader=test_dataloader,
            local_rank=self.local_rank,
            is_delay=self.config.is_delay,
            add_parity_check_bit=self.config.add_parity_check_bit,
        )


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model to count the parameters of.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameters(model):
    """
    Count the number of total parameters in a model.

    Args:
        model (torch.nn.Module): The model to count the parameters of.

    Returns:
        int: The total number of trainable parameters in the model.
    """

    return sum(p.numel() for p in model.parameters())


def calculate_BER(predictions, labels):
    """
    Calculate the bit error rate (BER) of the model's predictions.

    Args:
        predictions (torch.Tensor): The model's predictions.
        labels (torch.Tensor): The true labels.

    Returns:
        float: The bit error rate of the model's predictions.
    """
    return torch.sum(torch.abs(predictions - labels)).item() / labels.numel()


def calculate_BLER(predictions, labels):
    """
    Calculate the block error rate (BLER) of the model's predictions.

    Args:
        predictions (torch.Tensor): The model's predictions.
        labels (torch.Tensor): The true labels.

    Returns:
        float: The block error rate of the model's predictions.
    """
    return (
        torch.sum(torch.any(torch.abs(predictions - labels), dim=1)).item()
        / labels.shape[0]
    )


def calculate_sync_error(
    predicted_starting_point_probability, true_starting_point_onehot
):
    """
    Calculate the synchronization error between the predicted and true starting points.

    Args:
        predicted_starting_point_probability (torch.Tensor): The predicted starting point probabilities of shape (batch_size, dmax/ts+1).
        true_starting_point_onehot (torch.Tensor): The true starting point one-hot encoded tensor of shape (batch_size, dmax/ts+1).
        tp (float): The pulse shaping period.

    Returns:
        float: The synchronization error between the predicted and true starting points.
    """
    # The mean value of (true_starting_point_onehot - predicted_starting_point_probability)^2 / tp
    return torch.mean(
        torch.sum(
            (true_starting_point_onehot - predicted_starting_point_probability) ** 2,
            dim=1,
        )
    ).item()


def calculate_sync_error_rate(
    predicted_starting_point_probability, true_starting_point_onehot
):
    """
    Calculate the synchronization error rate between the predicted and true starting points.

    Args:
        predicted_starting_point_probability (torch.Tensor): The predicted starting point probabilities of shape (batch_size, n-n_M+1).
        true_starting_point_onehot (torch.Tensor): The true starting point one-hot encoded tensor of shape (batch_size, n-n_M+1).

    Returns:
        float: The synchronization error rate between the predicted and true starting points.
    """

    predicted_starting_indices = torch.argmax(
        predicted_starting_point_probability, dim=1
    )
    true_starting_indices = torch.argmax(true_starting_point_onehot, dim=1)

    # if predicted_starting_indices is larger than true_starting_indices+n_M, then it is considered as a sync error
    # if predicted_starting_indices is smaller than true_starting_indices-n_M, then it is considered as a sync error as well.
    sync_error = torch.sum(
        torch.abs(predicted_starting_indices - true_starting_indices) > 5
    ).item()
    sync_error_rate = sync_error / true_starting_indices.shape[0]
    return sync_error_rate


def unit_power_normalization(x, eps=1e-12):
    energy = torch.mean(
        torch.abs(x) ** 2, dim=2, keepdim=True
    )  # Shape: (batch_size, nb, 1)
    normalized_x = x / torch.sqrt(
        2 * energy + eps
    )  # Adding eps for numerical stability
    return normalized_x


# def setup_seed():
#     # Ensure deterministic behavior
#     torch.backends.cudnn.deterministic = True
#     random.seed(hash("setting random seeds") % 2**32 - 1)
#     np.random.seed(hash("improves reproducibility") % 2**32 - 1)
#     torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
#     torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
#     # add one for seed
#     # torch.cuda.manual_seed(42)
#     # torch.backends.cudnn.benchmark = False


def setup_seed(seed: int):
    # 1) Python built-ins
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) NumPy
    np.random.seed(seed)

    # 3) PyTorch CPU
    torch.manual_seed(seed)

    # 4) PyTorch GPU (if you’re using CUDA)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 5) cuDNN / cuBLAS determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # If you’re on PyTorch ≥1.8 you can also do:
    torch.use_deterministic_algorithms(True)

    # 6) For CUDA ≥11.3 you can further lock down non‐determinism:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def broadcast_seed(base_seed: int = None) -> int:
    """
    If base_seed is None, only rank 0 will pick a random seed;
    then we broadcast it to everyone.
    Returns the agreed-upon seed.
    """
    # make sure dist is initialized
    assert torch.distributed.is_initialized(), "call dist.init_process_group first"
    # if only one process, just use base_seed or random
    world_size = torch.distributed.get_world_size()
    print(f"world_size: {world_size}")
    if world_size <= 1:
        return base_seed if base_seed is not None else random.randint(0, 2**31 - 1)
    # multiple processes: rank 0 picks seed, others zero, then broadcast
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    if rank == 0:
        seed = base_seed if base_seed is not None else random.randint(0, 2**31 - 1)
    else:
        seed = 0
    print(f"rank {rank} seed: {seed}")
    seed_tensor = torch.tensor([seed], dtype=torch.long, device=device)
    print(f"rank {rank} seed_tensor: {seed_tensor}")
    # broadcast in-place from rank 0
    torch.distributed.broadcast(seed_tensor, src=0)
    print(f"rank {rank} seed_tensor after broadcast: {seed_tensor}")
    return seed_tensor.item()


class CustomLoss(nn.Module):
    def __init__(self, alpha=0.01):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.cce_loss = nn.CrossEntropyLoss()  # Categorical Cross-Entropy Loss

    def forward(self, bce_pred, bce_target, cce_pred, cce_target):
        """
        Compute the custom loss.

        Args:
            bce_pred (Tensor): Predictions for binary cross-entropy loss.
            bce_target (Tensor): Ground truth for binary cross-entropy loss.
            cce_pred (Tensor): Predictions for categorical cross-entropy loss (logits).
            cce_target (Tensor): Ground truth for categorical cross-entropy loss (class indices).

        Returns:
            Tensor: Total loss.
        """
        # Binary Cross-Entropy Loss
        loss_bce_ae = self.bce_loss(bce_pred, bce_target)

        # Categorical Cross-Entropy Loss
        loss_cce_det = self.cce_loss(
            cce_pred.to(torch.float32), cce_target.to(torch.float32)
        )

        # Total Loss
        loss_total = loss_bce_ae + self.alpha * loss_cce_det

        return loss_total


def add_even_parity_bit(x: torch.Tensor) -> torch.Tensor:
    """
    Adds an even parity check bit along dimension 1 of a binary tensor.

    Args:
        x (torch.Tensor): Binary tensor of shape (batch, k, 1) with values 0 or 1.

    Returns:
        torch.Tensor: Tensor of shape (batch, k+1, 1) with even parity bit appended.
    """
    # Ensure it's binary 0/1
    if not torch.all((x == 0) | (x == 1)):
        raise ValueError("Input tensor must contain only 0s and 1s.")

    # Sum along dimension 1 (keepdim=True to maintain shape for broadcasting)
    ones_count = x.sum(dim=1)  # shape: (batch, 1)

    # Even parity bit: 1 if sum is odd, else 0
    parity_bit = (ones_count % 2).to(x.dtype)  # 1 if odd, 0 if even

    # Append parity bit along dim=1
    parity_bit = parity_bit.unsqueeze(1)  # shape: (batch, 1, 1)
    return torch.cat([x, parity_bit], dim=1)


# ===== BLER Losses =====


def probs_to_logits(probs, eps=1e-10):
    # clamp to avoid log(0)
    probs = torch.clamp(probs, eps, 1 - eps)
    return torch.log(probs) - torch.log(1 - probs)


class ProductLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs, targets):
        logits = probs_to_logits(probs)
        term = (2 * targets.float() - 1) * logits
        return torch.mean(1.0 - torch.prod(torch.sigmoid(term), dim=1))


class MaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs, targets):
        bce = F.binary_cross_entropy_with_logits(
            probs, targets.float(), reduction="none"
        )
        return torch.mean(torch.max(bce, dim=1).values)


class SmoothMaxLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, probs, targets):
        bce = F.binary_cross_entropy_with_logits(
            probs, targets.float(), reduction="none"
        )
        weights = torch.exp(self.alpha * bce)
        return torch.mean(torch.sum(bce * weights, dim=1) / torch.sum(weights, dim=1))


class LogSumExpLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs, targets):
        bce = F.binary_cross_entropy_with_logits(
            probs, targets.float(), reduction="none"
        )
        K = bce.shape[1]
        return torch.mean(torch.log(torch.sum(torch.exp(bce), dim=1) - (K - 1)))


class PNormLoss(nn.Module):
    def __init__(self, p=2, gamma=1e-8):
        super().__init__()
        self.p = p
        self.gamma = gamma

    def forward(self, probs, targets):
        bce = F.binary_cross_entropy_with_logits(
            probs, targets.float(), reduction="none"
        )
        return torch.mean(
            (torch.sum(bce**self.p, dim=1) + self.gamma) ** (1.0 / self.p)
        )


class HybridBCESoftBLERLoss(nn.Module):
    """paper: DeepPolar+: Breaking the BER-BLER Trade-off with Self-Attention and SMART (SNR-MAtched Redundancy Technique) decoding

    Args:
        nn (_type_): _description_
    """

    def __init__(self, beta=1.0, eps=1e-12):
        super(HybridBCESoftBLERLoss, self).__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, probs, targets):
        probs = probs.squeeze(-1)
        targets = targets.squeeze(-1)

        # BCE term
        bce = F.binary_cross_entropy(probs, targets)

        # Soft-BLER term
        bit_likelihoods = probs * targets + (1 - probs) * (1 - targets)
        joint_prob = torch.prod(bit_likelihoods, dim=1)
        # take the log
        soft_bler = -torch.log(joint_prob + self.eps)
        # apply the mean
        soft_bler = soft_bler.mean()

        return bce + self.beta * soft_bler


class HybridBCEProductLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(HybridBCEProductLoss, self).__init__()
        self.beta = beta
        self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.product_loss = ProductLoss()

    def forward(self, probs, targets):
        # Binary Cross-Entropy Loss
        loss_bce = self.bce_loss(probs, targets)

        # Product Loss
        loss_product = self.product_loss(probs, targets)

        # Total Loss
        loss_total = loss_bce + self.beta * loss_product

        return loss_total


class HybridBCEMaxLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(HybridBCEMaxLoss, self).__init__()
        self.beta = beta
        self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.max_loss = MaxLoss()

    def forward(self, probs, targets):
        # Binary Cross-Entropy Loss
        loss_bce = self.bce_loss(probs, targets)
        # Max Loss
        loss_max = self.max_loss(probs, targets)
        # Total Loss
        loss_total = loss_bce + self.beta * loss_max

        return loss_total


class HybridBCESmoothMaxLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(HybridBCESmoothMaxLoss, self).__init__()
        self.beta = beta
        self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.smooth_max_loss = SmoothMaxLoss()

    def forward(self, probs, targets):
        # Binary Cross-Entropy Loss
        loss_bce = self.bce_loss(probs, targets)
        # Smooth Max Loss
        smooth_loss_max = self.smooth_max_loss(probs, targets)
        # Total Loss
        loss_total = loss_bce + self.beta * smooth_loss_max

        return loss_total


class HybridBCELogSumExpLoss(nn.Module):
    def __init__(self, beta=1.0):
        super(HybridBCELogSumExpLoss, self).__init__()
        self.beta = beta
        self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.log_sum_exp_loss = LogSumExpLoss()

    def forward(self, probs, targets):
        # Binary Cross-Entropy Loss
        loss_bce = self.bce_loss(probs, targets)
        # Log-Sum-Exp Loss
        loss_log_sum_exp = self.log_sum_exp_loss(probs, targets)
        # Total Loss
        loss_total = loss_bce + self.beta * loss_log_sum_exp

        return loss_total


class HybridBCEPNormLoss(nn.Module):
    def __init__(self, p, beta=1.0):
        super(HybridBCEPNormLoss, self).__init__()
        self.beta = beta
        self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.p_norm_loss = PNormLoss(p=p)

    def forward(self, probs, targets):
        # Binary Cross-Entropy Loss
        loss_bce = self.bce_loss(probs, targets)
        # P-Norm Loss
        loss_p_norm = self.p_norm_loss(probs, targets)
        # Total Loss
        loss_total = loss_bce + self.beta * loss_p_norm

        return loss_total
