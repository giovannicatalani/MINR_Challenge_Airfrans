import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# adapted from https://github.com/EmilienDupont/coinpp/blob/main/coinpp/metalearning.py and from https://github.com/LouisSerrano/coral


def graph_inner_loop(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_steps,
    inner_lr,
    is_train=False,
):
    """Performs inner loop, i.e. fits modulations such that the function
    representation can match the target features.


    """
    fitted_modulations = modulations
    for step in range(inner_steps):
        fitted_modulations = graph_inner_loop_step(
            func_rep,
            fitted_modulations,
            coords,
            features,
            batch_index,
            inner_lr,
            is_train,
        )

    return fitted_modulations


def graph_inner_loop_step(
    func_rep,
    modulations,
    coords,
    features,
    batch_index,
    inner_lr,
    is_train=False,
    last_element=False,
):
    """Performs a single inner loop step."""
    detach = False
    batch_size = modulations.shape[0]
    loss = 0
    with torch.enable_grad():
        # Note we multiply by batch size here to undo the averaging across batch
        # elements from the MSE function. Indeed, each set of modulations is fit
        # independently and the size of the gradient should not depend on how
        # many elements are in the batch

        features_recon = func_rep.modulated_forward(coords, modulations[batch_index])
        loss = ((features_recon - features) ** 2).mean() * batch_size

        # If we are training, we should create graph since we will need this to
        # compute second order gradients in the MAML outer loop
        grad = torch.autograd.grad(
            loss,
            modulations,
            create_graph=is_train and not detach,
        )[0]

    # Perform single gradient descent step
    return modulations - inner_lr * grad


def graph_outer_step(
    func_rep,
    graph,
    inner_steps,
    inner_lr,
    is_train=False,
    add_surf_loss=False,
    detach_modulations=False,
):
    """

    Args:
        coordinates (torch.Tensor): Shape (batch_size, *, coordinate_dim). Note this
            _must_ have a batch dimension.
        features (torch.Tensor): Shape (batch_size, *, feature_dim). Note this _must_
            have a batch dimension.
    """

    func_rep.zero_grad()
    batch_size = len(graph)
    if isinstance(func_rep, DDP):
        func_rep = func_rep.module

    modulations = torch.zeros_like(graph.modulations).requires_grad_()
    coords = graph.input
    features = graph.output
    surf = graph.surf

    # Run inner loop
    modulations = graph_inner_loop(
        func_rep,
        modulations,
        coords,
        features,
        graph.batch,
        inner_steps,
        inner_lr,
        is_train,
    )

    if detach_modulations:
        modulations = modulations.detach()  # 1er ordre

    loss = 0
    batch_size = modulations.shape[0]

    with torch.set_grad_enabled(is_train):
        features_recon = func_rep.modulated_forward(coords, modulations[graph.batch])
        loss = ((features_recon - features) ** 2).mean()
    if add_surf_loss:
        loss += 0.1 * ((features_recon[surf] - features[surf]) ** 2).mean()

    outputs = {
        "loss": loss,
        "modulations": modulations,
    }

    return outputs


# Function to extract latent vectors
def extract_latents(inr_in, loader, cfg, n_samples, batch_size=None, alpha_in=None):

    latent_dim = cfg.inr_in.latent_dim
    z = torch.zeros(n_samples, latent_dim)
    u = torch.zeros(n_samples, 2)

    # optim
    batch_size = cfg.optim.batch_size if batch_size == None else batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr_inr = cfg.optim.lr_inr
    gamma_step = cfg.optim.gamma_step
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    weight_decay_code = cfg.optim.weight_decay_code
    inner_steps = cfg.optim.inner_steps
    surf_loss = getattr(cfg.optim, "surf_loss", False)
    # alpha_in = nn.Parameter(torch.Tensor([lr_code]).cuda())

    fit_train_mse_in = 0

    current_sample_index = 0
    for substep, graph in enumerate(loader):
        n_batch_sample = len(graph)
        inr_in.train()

        graph.modulations = torch.zeros(batch_size, latent_dim)
        graph = graph.cuda()

        outputs = graph_outer_step(
            inr_in,
            graph,
            inner_steps,
            alpha_in,
            is_train=False,
            add_surf_loss=surf_loss,
        )

        loss = outputs["loss"].cpu().detach()
        fit_train_mse_in += loss.item() * n_batch_sample
        z0 = outputs["modulations"].detach()
        z[current_sample_index : current_sample_index + n_batch_sample] = z0[
            :n_batch_sample
        ]
        u[current_sample_index : current_sample_index + n_batch_sample] = (
            graph.cond.cpu()
        )

        current_sample_index += n_batch_sample

    train_loss_in = fit_train_mse_in / (n_samples)

    print(train_loss_in)

    return z, u
