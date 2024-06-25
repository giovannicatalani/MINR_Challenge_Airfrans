import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parents[1]))
import numpy as np
import torch
import torch.nn as nn

# from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset
from utils.load_models import create_inr_instance, load_inr, load_processor
from utils.metalearning import graph_outer_step as outer_step, extract_latents
from utils.dataset import process_dataset_field, Struct, subsample_dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import copy


def train_inr(cfg, train_dataset, val_dataset, results_directory):
    # Convert cfg to OmegaConf object if it's a dictionary
    if isinstance(cfg, dict):
        # cfg = OmegaConf.create(cfg)
        cfg = Struct(cfg)

    # Accessing parameters using dot notation
    batch_size = cfg.optim.batch_size
    epochs = cfg.optim.epochs
    inner_steps = cfg.optim.inner_steps
    lr_inr = cfg.optim.lr_inr
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    input_dim = cfg.inr_in.input_dim
    output_dim = cfg.inr_in.output_dim
    latent_dim = cfg.inr_in.latent_dim
    target_field = cfg.inr_in.target_field
    num_points = None if target_field in ["normals", "p_surf"] else cfg.optim.num_points
    surf_loss = getattr(cfg.optim, "surf_loss", False)

    # Create INR instance
    cfg.inr = cfg.inr_in
    inr_in = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device="cuda"
    )
    alpha_in = nn.Parameter(torch.Tensor([lr_code]).cuda())

    optimizer_in = torch.optim.AdamW(
        [
            {"params": inr_in.parameters()},
            {"params": alpha_in, "lr": meta_lr_code, "weight_decay": 0},
        ],
        lr=lr_inr,
        weight_decay=0,
    )

    # Generate run name
    run_name = f"inr_{target_field}"

    best_loss = np.inf
    train_loss_history, test_loss_history = [], []

    for step in tqdm(range(epochs), desc="Training INR"):
        train_loss_in, test_loss_in = 0, 0
        fit_train_mse_in, fit_test_mse_in = 0, 0
        use_rel_loss = step % 10 == 0

        if target_field in ["normals", "p_surf"]:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
        else:
            # Dynamic subsampling for training
            train_loader = DataLoader(
                subsample_dataset(train_dataset, num_points),
                batch_size=batch_size,
                shuffle=True,
            )

        # Training loop
        for graph in train_loader:
            n_samples = graph.num_graphs
            inr_in.train()
            graph.to("cuda")
            graph.modulations = torch.zeros(batch_size, latent_dim, device="cuda")

            outputs = outer_step(
                inr_in,
                graph,
                inner_steps,
                alpha_in,
                is_train=True,
                add_surf_loss=surf_loss,
            )

            optimizer_in.zero_grad()
            outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr_in.parameters(), clip_value=1.0)
            optimizer_in.step()
            loss = outputs["loss"].cpu().detach()
            fit_train_mse_in += loss.item() * n_samples
            z0 = outputs["modulations"].detach()

        train_loss_in = fit_train_mse_in / len(train_dataset)
        train_loss_history.append({"epoch": step, "loss": train_loss_in})
        print("Train Loss", train_loss_in)

        # Validation loop
        if step % 5 == 0:  # Adjust as per your validation frequency

            if target_field in ["normals", "p_surf"]:
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=True
                )
            else:
                # Dynamic subsampling for training
                val_loader = DataLoader(
                    subsample_dataset(val_dataset, num_points),
                    batch_size=batch_size,
                    shuffle=True,
                )

            for substep, graph in enumerate(val_loader):
                n_samples = len(graph)
                inr_in.train()

                graph.modulations = torch.zeros(batch_size, latent_dim)
                graph = graph.cuda()

                outputs = outer_step(
                    inr_in,
                    graph,
                    inner_steps,
                    alpha_in,
                    is_train=False,
                    add_surf_loss=surf_loss,
                )
                loss = outputs["loss"].cpu().detach()
                fit_test_mse_in += loss.item() * n_samples
                z0 = outputs["modulations"].detach()

            test_loss_in = fit_test_mse_in / len(val_dataset)
            test_loss_history.append({"epoch": step, "loss": test_loss_in})
            print("Test Loss", test_loss_in)

            """
            plt.figure()
            plt.plot([p['epoch'] for p in train_loss_history], [p['loss'] for p in train_loss_history], label='Train Loss')
            plt.plot([p['epoch'] for p in test_loss_history], [p['loss'] for p in test_loss_history], label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.title('Training and Test Loss')
            plt.legend()
            plt.savefig(os.path.join(results_directory, f"{run_name}_loss_plot.png"))
            plt.close()
            """

            if test_loss_in < best_loss:
                best_loss = test_loss_in
                best_model_state = {
                    "cfg": cfg,
                    "epoch": step,
                    "inr_in": inr_in.state_dict(),
                    "optimizer_inr_in": optimizer_in.state_dict(),
                    "alpha_in": alpha_in,
                }

    return best_model_state


def train_regression(
    cfg, z_train_input, z_val_input, z_train_output, z_val_output, results_directory
):

    # Convert cfg to OmegaConf object if it's a dictionary
    if isinstance(cfg, dict):
        # cfg = OmegaConf.create(cfg)
        cfg = Struct(cfg)

    model_save_path = os.path.join(results_directory, "best_model.pth")

    input_dim = z_train_input.shape[1]  # Dimension of concatenated input latent vectors
    output_dim = z_train_output.shape[
        1
    ]  # Dimension of concatenated output latent vectors

    model = load_processor(
        cfg, input_dim, output_dim, model_save_path=model_save_path
    ).cuda()
    # Define optimizer
    optimizer_pred = torch.optim.AdamW(
        model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay
    )

    train_dataset = TensorDataset(z_train_input, z_train_output)
    val_dataset = TensorDataset(z_val_input, z_val_output)
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.model.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.model.batch_size, shuffle=False
    )

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    for epoch in tqdm(range(cfg.model.epochs)):
        model.train()
        train_loss = 0.0
        for z_input, z_output in train_loader:
            z_input, z_output = z_input.cuda(), z_output.cuda()
            optimizer_pred.zero_grad()
            z_pred = model(z_input)
            loss = ((z_pred - z_output) ** 2).mean()
            loss.backward()
            optimizer_pred.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z_input, z_output in val_loader:
                z_input, z_output = z_input.cuda(), z_output.cuda()
                z_pred = model(z_input)
                loss = ((z_pred - z_output) ** 2).mean()
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_save_path)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss}, Val Loss: {val_loss}")
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_directory, 'regression_loss_plot.png'))
    plt.close()
    """

    # Reload the best model state load_processor(cfg, input_dim, output_dim, model_save_path=None)
    best_model = load_processor(cfg, input_dim, output_dim, model_save_path=None).cuda()
    best_model.load_state_dict(best_model_state)

    return best_model


def global_train(cfg, target_fields, train_dataset, val_dataset, trainings_dir):
    z_train_input_list, z_val_input_list = [], []
    z_train_output_list, z_val_output_list = [], []

    model_dict = {}

    for target_field in target_fields:
        print(f"Processing target field: {target_field}")

        # Here add process_dataset_function
        try:
            global_coef_norm
        except NameError:
            train_dataset_field, global_coef_norm = process_dataset_field(
                train_dataset, target_field, num_points=500, global_coef_norm=None
            )
        else:
            train_dataset_field, _ = process_dataset_field(
                train_dataset,
                target_field,
                num_points=None,
                global_coef_norm=global_coef_norm,
            )

        val_dataset_field, _ = process_dataset_field(
            val_dataset,
            target_field,
            num_points=500,
            global_coef_norm=global_coef_norm,
        )

        # Update cfg for the current target field
        cfg_inr = cfg[target_field]
        model_path = os.path.join(trainings_dir, f"inr_{target_field}.pt")

        input_dim = cfg_inr["inr_in"]["input_dim"]
        output_dim = cfg_inr["inr_in"]["output_dim"]

        # Check if a pre-trained model exists
        if os.path.exists(model_path):
            print(f"Loading pre-trained model for {target_field}")
            trained_inr_state = torch.load(model_path)
        else:
            # Train INR model for the current target field
            trained_inr_state = train_inr(
                cfg_inr, train_dataset_field, val_dataset_field, trainings_dir
            )
            torch.save(trained_inr_state, model_path)  # Save the trained model

        cfg_single = trained_inr_state["cfg"]
        inr_weights = trained_inr_state["inr_in"]
        alpha_in = trained_inr_state["alpha_in"]

        # Load model and store all in a dictionary
        inr_in = load_inr(inr_weights, cfg_single, input_dim, output_dim)
        model_dict[target_field] = {
            "trained_model": inr_in,
            "cfg": cfg_single,
            "alpha_in": alpha_in,
        }

        # Prepare DataLoaders
        batch_size = 1  # Batch_size not too big to fit in memory full graphs

        train_loader = DataLoader(
            dataset=train_dataset_field, batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            dataset=val_dataset_field, batch_size=batch_size, shuffle=False
        )

        ntrain = len(train_dataset_field)
        nval = len(val_dataset_field)

        # Extract latents
        z_train_inr, train_cond = extract_latents(
            inr_in,
            train_loader,
            cfg_single,
            ntrain,
            alpha_in=alpha_in,
            batch_size=batch_size,
        )
        z_val_inr, val_cond = extract_latents(
            inr_in,
            val_loader,
            cfg_single,
            nval,
            alpha_in=alpha_in,
            batch_size=batch_size,
        )

        # Concatenate latents based on INR type (input or output)
        if cfg_inr["inr_in"]["is_input"]:
            z_train_input_list.append(z_train_inr)
            z_val_input_list.append(z_val_inr)
        else:
            z_train_output_list.append(z_train_inr)
            z_val_output_list.append(z_val_inr)

    # Concatenate all input and output latents
    z_train_input = torch.cat(z_train_input_list + [train_cond], dim=1)
    z_val_input = torch.cat(z_val_input_list + [val_cond], dim=1)

    z_train_output = torch.cat(z_train_output_list, dim=1)
    z_val_output = torch.cat(z_val_output_list, dim=1)

    # Compute mean and standard deviation for each component in training latent vectors
    mean_input = z_train_input.mean(dim=0)
    std_input = z_train_input.std(dim=0)

    mean_output = z_train_output.mean(dim=0)
    std_output = z_train_output.std(dim=0)

    # Normalize validation and test latent vectors
    z_train_input = (z_train_input - mean_input) / (std_input + 1e-8)
    z_val_input = (z_val_input - mean_input) / (std_input + 1e-8)

    z_train_output = (z_train_output - mean_output) / (std_output + 1e-8)
    z_val_output = (z_val_output - mean_output) / (std_output + 1e-8)

    z_stats = {
        "mean_input": mean_input,
        "std_input": std_input,
        "mean_output": mean_output,
        "std_output": std_output,
    }

    dict_inputs = {
        "z_train_input": z_train_input,
        "z_val_input": z_val_input,
        "z_train_cond": train_cond,
        "z_val_cond": val_cond,
        "z_train_output": z_train_output,
        "z_val_output": z_val_output,
        "trainings_dir": trainings_dir,
    }

    torch.save(dict_inputs, os.path.join(trainings_dir, "dict_inputs.pt"))

    model_dict["regression"] = train_regression(
        cfg["regression"],
        z_train_input,
        z_val_input,
        z_train_output,
        z_val_output,
        trainings_dir,
    )

    return model_dict, z_stats, global_coef_norm


def predict_test(
    cfg, model_dict, test_dataset, z_stats, global_coef_norm, all_outputs=False
):

    z_test_input_list = []
    target_fields_input = ["normals", "implicit_distance"]
    target_fields_output = ["all_outputs"] if all_outputs else ["p", "Ux", "Uy", "nut"]
    ntest = len(test_dataset)

    for target_field in target_fields_input:
        print(f"Processing target field: {target_field}")
        cfg_single = model_dict[target_field]["cfg"]
        # Convert cfg to OmegaConf object if it's a dictionary
        if isinstance(cfg_single, dict):
            # cfg_single = OmegaConf.create(cfg_single)
            cfg_single = Struct(cfg_single)

        test_dataset_field, _ = process_dataset_field(
            test_dataset,
            target_field,
            num_points=None,
            global_coef_norm=global_coef_norm,
        )

        # Prepare DataLoaders
        batch_size = 4  # Batch_size not too big to fit in memory full graphs
        test_loader = DataLoader(
            test_dataset_field, batch_size=batch_size, shuffle=False
        )
        z_test_inr, test_cond = extract_latents(
            model_dict[target_field]["trained_model"],
            test_loader,
            cfg_single,
            ntest,
            batch_size=batch_size,
            alpha_in=model_dict[target_field]["alpha_in"],
        )

        # Concatenate latents based on INR type (input or output)
        z_test_input_list.append(z_test_inr)

    # Concatenate all input and output latents
    z_test_input = torch.cat(z_test_input_list + [test_cond], dim=1)
    z_test_input = (z_test_input - z_stats["mean_input"]) / (
        z_stats["std_input"] + 1e-8
    )

    model_dict["regression"].eval()

    # Predict latents for test dataset
    with torch.no_grad():
        z_pred_test = torch.zeros(len(z_test_input), z_stats["mean_output"].shape[0])
        for i, z_input in enumerate(
            torch.utils.data.DataLoader(z_test_input, batch_size=batch_size)
        ):
            z_input = z_input.cuda()
            z_pred_test[i * batch_size : (i + 1) * batch_size] = model_dict[
                "regression"
            ](z_input).cpu()

    # Renormalize the predicted latents
    z_pred_test_renorm = z_pred_test * z_stats["std_output"] + z_stats["mean_output"]

    # Predict outputs for test dataset
    result_test = {"predictions": {}, "targets": {}, "mse": {}}
    idx = 0

    for target_field in target_fields_output:

        cfg_single = cfg[target_field]
        test_dataset_field, _ = process_dataset_field(
            test_dataset,
            target_field,
            num_points=None,
            global_coef_norm=global_coef_norm,
        )

        # Convert cfg to OmegaConf object if it's a dictionary
        if isinstance(cfg_single, dict):
            # cfg_single = OmegaConf.create(cfg_single)
            cfg_single = Struct(cfg_single)

        latent_out = z_pred_test_renorm[
            :, idx : idx + cfg_single.inr_in.latent_dim
        ].cuda()
        idx += cfg_single.inr_in.latent_dim

        inr_model = model_dict[target_field]["trained_model"]
        inr_model.eval()
        result_test["predictions"][target_field] = []
        result_test["targets"][target_field] = []
        result_test["mse"][target_field] = []

        for substep, graph in enumerate(test_dataset_field):
            pred = (
                inr_model.modulated_forward(
                    graph.input.cuda(), latent_out[substep].unsqueeze(0)
                )
                .detach()
                .cpu()
            )
            target = graph.output
            mse = ((pred.numpy() - target.numpy()) ** 2).mean(axis=0)
            result_test["predictions"][target_field].append(pred.numpy())
            result_test["targets"][target_field].append(target.numpy())
            result_test["mse"][target_field].append(mse)

    return result_test
