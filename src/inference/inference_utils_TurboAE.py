"""
This files contains the gengeral-purpose functions to train the model: Turbo AutoEncoder.

"""

import numpy as np
import torch

import wandb


def inference_loop(
    config, model, test_dataloader, local_rank, is_delay, add_parity_check_bit
):
    """
    inference loop for the well trained model.


    """

    print("Starting inference ...")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # inference snr_db values is a list from test_snr_min to test_snr_max with step 2, contains 2 sides of the SNR_db
    inference_snr_db = np.arange(config.test_snr_low, config.test_snr_high + 0.5, 2)

    # initialize the list to store the bit error rate and block error rate
    infer_ber_list = []  # list to store the bit error rate
    infer_bler_list = []  # list to store the block error rate

    # switch to evaluation mode
    model.eval()

    # turn off gradients
    with torch.no_grad():
        for i in range(len(inference_snr_db)):
            # initialize the list to store the bit error rate and block error rate
            ber_snr_i = []  # list to store the bit error rate
            bler_snr_i = []  # list to store the block error rate

            # Initialize the result dictionary
            bit_error = 64
            error_distribution = {i: 0 for i in range(bit_error + 1)}
            error_probabilities_gt1_list = []
            error_probabilities_gt0_list = []
            error_indices_list = []

            for batch_index, data in enumerate(test_dataloader, 1):
                # get the features and targets, X has shape (batch_size, 1, k)
                X = data.to(local_rank)

                if config.channel == "AWGN":
                    # forward pass, prediction by the model, shape (batch_size, 1, k)
                    predictions = model.forward(
                        X,
                        SNR_db_low=inference_snr_db[i],
                        SNR_db_high=inference_snr_db[i],
                    )
                    predictions = torch.clamp(predictions, 0.0, 1.0)
                    # round the predictions to 0 or 1
                    predictions_round = torch.round(predictions)
                elif config.channel == "proakis_c_channel" and is_delay:
                    # forward pass, prediction by the model, shape (batch_size, 1, k)
                    predictions = model.forward(
                        X,
                        SNR_db_low=inference_snr_db[i],
                        SNR_db_high=inference_snr_db[i],
                    )
                    predictions = torch.clamp(predictions, 0.0, 1.0)
                    # round the predictions to 0 or 1
                    predictions_round = torch.round(predictions)

                    if add_parity_check_bit:
                        # even parity check of predictions_round, shape (batch_size, k+1, 1)
                        predictions_round_parity_check = has_even_number_of_ones(
                            predictions_round
                        )
                        updated_predictions = conditional_min_flip(
                            predictions, predictions_round_parity_check
                        )
                        updated_predictions_round = torch.round(updated_predictions)
                        # cut off the parity check bit,
                        predictions_round = updated_predictions_round[:, :-1, :]
                        # X = X[:, :, :-1]

                # compute the bit error rate and block error rate
                infer_ber_batch, infer_bler_batch = compute_ber_bler(
                    predictions_round, X
                )

                stats = calculate_bit_error_statistics(
                    predictions[:, :-1, :], predictions_round, X, bit_error
                )
                # update the error distribution
                for key, value in stats["error_distribution"].items():
                    error_distribution[key] += value

                error_probabilities_gt1_list.extend(stats["error_probabilities_gt1"])
                error_probabilities_gt0_list.extend(stats["error_probabilities_gt0"])
                error_indices_list.extend(stats["error_indices"])

                # append the bit error rate and block error rate to the list
                ber_snr_i.append(infer_ber_batch)
                bler_snr_i.append(infer_bler_batch)

            # compute the average bit error rate and block error rate
            infer_ber_avg = sum(ber_snr_i) / len(ber_snr_i)
            infer_bler_avg = sum(bler_snr_i) / len(bler_snr_i)

            # print
            print(
                f"For SNR = {inference_snr_db[i]}, the average BER is {infer_ber_avg}, the average BLER is {infer_bler_avg}"
            )

            # append the average bit error rate and block error rate to the list
            infer_ber_list.append(infer_ber_avg)
            infer_bler_list.append(infer_bler_avg)

    # # bar plot for error_distribution using matplotlib
    # import matplotlib.pyplot as plt

    # # set the size of the figure
    # plt.figure(figsize=(40, 10))
    # plt.bar(error_distribution.keys(), error_distribution.values())
    # plt.xlabel("Number of erroneous bits", fontsize=20)
    # plt.ylabel("Count", fontsize=20)
    # plt.title(
    #     f"Distribution of number of erroneous bits ($2e5$ samples in total) - 20 dB",
    #     fontsize=20,
    # )
    # plt.xticks(list(error_distribution.keys()), fontsize=15)

    # # add percentage values
    # total_samples = sum(error_distribution.values())
    # # rotate the text for better visibility
    # for i in range(len(error_distribution)):
    #     plt.text(
    #         i,
    #         error_distribution[list(error_distribution.keys())[i]],
    #         f"{error_distribution[list(error_distribution.keys())[i]] / total_samples * 100:.3f}%",
    #         ha="center",
    #         va="bottom",
    #         rotation=90,
    #     )
    # plt.savefig("error_distribution_20db.png", bbox_inches="tight", dpi=500)

    # define our custom x axis metric
    wandb.define_metric("custom_step")

    # define which metrics will be plotted against it
    # first plot for BER
    wandb.define_metric(
        "BER v.s. SNR (AWGN)",
        step_metric="custom_step",
    )

    # second plot for BLER
    wandb.define_metric(
        "BLER v.s. SNR (AWGN)",
        step_metric="custom_step",
    )

    for i in range(len(inference_snr_db)):
        log_dict = {
            "BER v.s. SNR (AWGN)": infer_ber_list[i],
            "custom_step": inference_snr_db[i],
        }
        wandb.log(log_dict)

        log_dict = {
            "BLER v.s. SNR (AWGN)": infer_bler_list[i],
            "custom_step": inference_snr_db[i],
        }
        wandb.log(log_dict)

    print("The snr_db values for inference are: ", inference_snr_db)
    print(
        "The average bit error rate is: ",
        [round(tensor.item(), 7) for tensor in infer_ber_list],
    )
    print(
        "The average block error rate is: ",
        [round(tensor.item(), 7) for tensor in infer_bler_list],
    )

    ############################################################################################################
    # custom chart
    # Create a table with the columns to plot
    data_ber = []
    data_bler = []

    for i in range(len(infer_ber_list)):
        data_ber.append([inference_snr_db[i], infer_ber_list[i].item()])

    for i in range(len(infer_bler_list)):
        data_bler.append([inference_snr_db[i], infer_bler_list[i].item()])

    table_ber = wandb.Table(data=data_ber, columns=["SNR", "BER"])
    table_bler = wandb.Table(data=data_bler, columns=["SNR", "BLER"])

    # Map from the table's columns to the chart's fields
    fields_ber = {"x": "SNR", "y": "BER"}
    fields_bler = {"x": "SNR", "y": "BLER"}

    # Use the table to populate various custom charts
    # ber_plot = wandb.plot.line(table, x="SNR", y="BER", title="BER vs. SNR")
    # bler_plot = wandb.plot.line(table, x="SNR", y="BLER", title="BLER vs. SNR")
    if config.channel == "AWGN":
        ber_plot = wandb.plot_table(
            vega_spec_name="BER_new", data_table=table_ber, fields=fields_ber
        )
        bler_plot = wandb.plot_table(
            vega_spec_name="BLER_new", data_table=table_bler, fields=fields_bler
        )
        # Log custom tables, which will show up in customizable charts in the UI
        wandb.log({"BER": ber_plot, "BLER": bler_plot})
    elif config.channel == "proakis_c_channel":
        ber_plot = wandb.plot_table(
            vega_spec_name="BER_BlockFadingChannel",
            data_table=table_ber,
            fields=fields_ber,
        )
        bler_plot = wandb.plot_table(
            vega_spec_name="BLER_BlockFadingChannel",
            data_table=table_bler,
            fields=fields_bler,
        )
        # Log custom tables, which will show up in customizable charts in the UI
        wandb.log(
            {"BER_BlockFadingChannel": ber_plot, "BLER_BlockFadingChannel": bler_plot}
        )
    else:
        raise ValueError("Invalid channel type")

    print("Finished inference!")

    return model


def compute_ber_bler(predictions, targets):
    """
    Compute the bit error rate and block error rate.

    Args:
        predictions (torch.Tensor): The predicted values of shape (batch_size, k, 1).
        targets (torch.Tensor): The target values of shape (batch_size, k, 1).

    Returns:
        float: The bit error rate.
        float: The block error rate.
    """
    # compute the bit error rate
    ber = torch.sum(torch.abs(predictions - targets)) / (
        targets.shape[0] * targets.shape[1]
    )

    # compute the block error rate
    bler = (
        torch.sum(torch.any(torch.abs(predictions - targets), dim=1)) / targets.shape[0]
    )

    return ber, bler


def calculate_bit_error_statistics(prediction_raw, predictions_rounds, X, bit_error):
    """
    Extended version that also separates raw probabilities for wrong bits
    by the ground-truth value (0 / 1).

    Args
    ----
    prediction_raw      : Float tensor (batch, k, 1) - probability of bit == 1
    predictions_rounds  : Binary tensor (batch, k, 1) - thresholded version
    X                   : Binary tensor (batch, k, 1) - ground-truth bits
    bit_error           : int - max #bit errors to appear in histogram

    Returns
    -------
    dict with keys
        error_distribution      : {0 … bit_error : count}
        error_probabilities_gt1 : [float]  # wrong bits, ground truth == 1
        error_probabilities_gt0 : [float]  # wrong bits, ground truth == 0
        error_indices           : [bit_idx]
    """

    # 1) ensure integer 0/1 for “boolean” tensors
    predictions_rounds = predictions_rounds.int()
    X = X.int()

    # 2) mask of wrong decisions (XOR)
    error_mask = predictions_rounds != X  # shape (batch, k, 1)

    # 3) #wrong bits per sample  →  histogram
    errors_per_sample = error_mask.squeeze(-1).sum(dim=1)  # (batch,)
    error_distribution = {
        i: (errors_per_sample == i).sum().item() for i in range(bit_error + 1)
    }

    # ------------------------------------------------------------------
    # 4) split wrong-bit probabilities by the GT value in X
    # ------------------------------------------------------------------
    # masks
    wrong_and_gt1 = error_mask & (X == 1)
    wrong_and_gt0 = error_mask & (X == 0)

    error_probabilities_gt1 = prediction_raw[wrong_and_gt1].tolist()
    error_probabilities_gt0 = prediction_raw[wrong_and_gt0].tolist()

    # ------------------------------------------------------------------
    # 5) coordinates (sample, bit) for *all* wrong bits
    # ------------------------------------------------------------------
    wrong_coords = torch.nonzero(error_mask.squeeze(-1), as_tuple=False)
    error_indices = [int(b) for s, b in wrong_coords]

    return {
        "error_distribution": error_distribution,
        "error_probabilities_gt1": error_probabilities_gt1,
        "error_probabilities_gt0": error_probabilities_gt0,
        "error_indices": error_indices,
    }


def has_even_number_of_ones(x: torch.Tensor) -> torch.Tensor:
    """
    Checks if each sample in the batch has an even number of ones along the last dimension.

    Args:
        x (torch.Tensor): Binary tensor of shape (batch, k+1, 1) with values 0 or 1.

    Returns:
        torch.Tensor: Bool tensor of shape (batch, 1), True if even number of ones, False if odd.
    """
    if not torch.all((x == 0) | (x == 1)):
        raise ValueError("Input tensor must contain only 0s and 1s.")

    # Count ones along the last dimension (dim=2)
    ones_count = x.sum(dim=1)  # shape: (batch, 1)

    # Even number check → remainder == 0
    return ones_count % 2 == 0


def conditional_min_flip(prediction: torch.Tensor, check: torch.Tensor):
    """
    If check[b] == True: keep prediction[b] unchanged.
    If check[b] == False:
        •	Find the element in prediction[b, :, 0] closest to 0.5.
        •	Replace that element with 1 - value.
        •	Leave all other elements unchanged.

    Args:
        prediction: Tensor of shape (batch, k+1, 1), values in [0,1]
        check: Bool tensor of shape (batch, 1)

    Returns:
        Modified tensor of shape (batch, k+1, 1)
    """
    output = prediction.clone()
    batch, length, _ = prediction.shape

    # Find index of value closest to 0.5 for each batch
    vals = prediction.squeeze(-1)  # (batch, k+1)
    idx = torch.argmin(torch.abs(vals - 0.5), dim=1)  # (batch,)

    # Create index grid for scatter update
    batch_idx = torch.arange(batch, device=prediction.device)

    # Only flip for check == False
    mask = ~check.view(-1)  # (batch,)

    # Gather original values to flip
    chosen_vals = vals[batch_idx, idx]  # (batch,)
    flipped_vals = 1.0 - chosen_vals

    # Assign back only for masked batches
    output[batch_idx[mask], idx[mask], 0] = flipped_vals[mask]

    return output
