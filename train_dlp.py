"""
Single-GPU training of DLPv2
"""
# imports
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
import argparse
# torch
import torch
import torch.nn.functional as F
from utils.loss_functions import calc_reconstruction_loss, VGGDistance
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.optim as optim
# modules
from models import ObjectDLP
# datasets
from datasets.get_dataset import get_image_dataset
# util functions
from utils.util_func import plot_keypoints_on_image_batch, prepare_logdir, save_config, log_line, \
    plot_bb_on_image_batch_from_z_scale_nms, plot_bb_on_image_batch_from_masks_nms, get_config
from eval.eval_model import evaluate_validation_elbo
from eval.eval_gen_metrics import eval_dlp_im_metric

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_dlp(config_path='./configs/shapes.json'):
    # load config
    try:
        config = get_config(config_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")
    hparams = config  # to save a copy of the hyper-parameters
    # data and general
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root
    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['num_epochs']
    topk = min(config['topk'], config['n_kp_enc'])  # top-k particles to plot
    eval_epoch_freq = config['eval_epoch_freq']
    weight_decay = config['weight_decay']
    iou_thresh = config['iou_thresh']  # threshold for NMS for plotting bounding boxes
    run_prefix = config['run_prefix']
    load_model = config['load_model']
    pretrained_path = config['pretrained_path']  # path of pretrained model to load, if None, train from scratch
    adam_betas = config['adam_betas']
    adam_eps = config['adam_eps']
    scheduler_gamma = config['scheduler_gamma']
    eval_im_metrics = config['eval_im_metrics']
    device = config['device']
    if 'cuda' in device:
        device = torch.device(f'{device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    # model
    kp_range = config['kp_range']
    kp_activation = config['kp_activation']
    enc_channels = config['enc_channels']
    prior_channels = config['prior_channels']
    pad_mode = config['pad_mode']
    n_kp = config['n_kp']  # kp per patch in prior, best to leave at 1
    n_kp_prior = config['n_kp_prior']  # number of prior kp to filter for the kl
    n_kp_enc = config['n_kp_enc']  # total posterior kp
    patch_size = config['patch_size']  # prior patch size
    anchor_s = config['anchor_s']  # posterior patch/glimpse ratio of image size
    learned_feature_dim = config['learned_feature_dim']
    dropout = config['dropout']
    use_resblock = config['use_resblock']
    use_correlation_heatmaps = config['use_correlation_heatmaps']  # use heatmaps for tracking
    use_tracking = config['use_tracking']
    enable_enc_attn = config['enable_enc_attn']  # enable attention between patches in the particle encoder
    filtering_heuristic = config["filtering_heuristic"]  # filtering heuristic to filter prior keypoints

    # optimization
    warmup_epoch = config['warmup_epoch']
    recon_loss_type = config['recon_loss_type']
    beta_kl = config['beta_kl']
    beta_rec = config['beta_rec']
    kl_balance = config['kl_balance']  # balance between visual features and the other particle attributes
    train_enc_prior = config['train_enc_prior']

    # priors
    sigma = config['sigma']  # std for constant kp prior, leave at 1 for deterministic chamfer-kl
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']  # transparency beta distribution "a"
    obj_on_beta = config['obj_on_beta']  # transparency beta distribution "b"

    # load data
    dataset = get_image_dataset(ds, root, mode='train', image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                            drop_last=True)
    # model
    model = ObjectDLP(cdim=ch, enc_channels=enc_channels, prior_channels=prior_channels,
                      image_size=image_size, n_kp=n_kp, learned_feature_dim=learned_feature_dim,
                      pad_mode=pad_mode, sigma=sigma,
                      dropout=dropout, patch_size=patch_size, n_kp_enc=n_kp_enc,
                      n_kp_prior=n_kp_prior, kp_range=kp_range, kp_activation=kp_activation,
                      anchor_s=anchor_s, use_resblock=use_resblock,
                      scale_std=scale_std, offset_std=offset_std, obj_on_alpha=obj_on_alpha,
                      obj_on_beta=obj_on_beta,
                      use_correlation_heatmaps=use_correlation_heatmaps, use_tracking=use_tracking,
                      enable_enc_attn=enable_enc_attn, filtering_heuristic=filtering_heuristic).to(device)
    print(model.info())
    # prepare saving location
    run_name = f'{ds}_dlp' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')
    save_config(log_dir, hparams)

    # prepare loss functions
    if recon_loss_type == "vgg":
        recon_loss_func = VGGDistance(device=device)
    else:
        recon_loss_func = calc_reconstruction_loss

    # optimizer and scheduler
    optimizer = optim.Adam(model.get_parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma, verbose=True)

    if load_model and pretrained_path is not None:
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
            print("loaded model from checkpoint")
        except:
            print("model checkpoint not found")

    # log statistics
    losses = []
    losses_rec = []
    losses_kl = []
    losses_kl_kp = []
    losses_kl_feat = []
    losses_kl_scale = []
    losses_kl_depth = []
    losses_kl_obj_on = []

    # initialize validation statistics
    valid_loss = best_valid_loss = 1e8
    valid_losses = []
    best_valid_epoch = 0

    # save PSNR values of the reconstruction
    psnrs = []

    # image metrics
    if eval_im_metrics:
        val_lpipss = []
        best_val_lpips_epoch = 0
        val_lpips = best_val_lpips = 1e8

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        batch_losses_rec = []
        batch_losses_kl = []
        batch_losses_kl_kp = []
        batch_losses_kl_feat = []
        batch_losses_kl_scale = []
        batch_losses_kl_depth = []
        batch_losses_kl_obj_on = []
        batch_psnrs = []

        pbar = tqdm(iterable=dataloader)
        for batch in pbar:
            x = batch[0].to(device)

            if len(x.shape) == 5 and not use_tracking:
                # [bs, T, ch, h, w]
                x = x.view(-1, *x.shape[2:])
            elif len(x.shape) == 4 and use_tracking:
                # [bs, ch, h, w]
                x = x.unsqueeze(1)
            x_prior = x  # the input image to the prior is the same as the posterior
            # noisy = (epoch < (warmup_epoch + 1))
            noisy = False
            # forward pass
            model_output = model(x, x_prior=x_prior, warmup=(epoch < warmup_epoch), noisy=noisy, bg_masks_from_fg=False,
                                 train_enc_prior=train_enc_prior)
            # calculate loss
            all_losses = model.calc_elbo(x, model_output, warmup=(epoch < warmup_epoch), beta_kl=beta_kl,
                                         beta_rec=beta_rec, kl_balance=kl_balance,
                                         recon_loss_type=recon_loss_type,
                                         recon_loss_func=recon_loss_func, noisy=noisy)
            loss = all_losses['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # output for logging and plotting
            mu_p = model_output['kp_p']
            z_base = model_output['z_base']
            mu_offset = model_output['mu_offset']
            logvar_offset = model_output['logvar_offset']
            rec_x = model_output['rec']
            mu_scale = model_output['mu_scale']
            mu_depth = model_output['mu_depth']
            # object stuff
            dec_objects_original = model_output['dec_objects_original']
            cropped_objects_original = model_output['cropped_objects_original']
            obj_on = model_output['obj_on']  # [batch_size, n_kp]
            alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]

            psnr = all_losses['psnr']
            obj_on_l1 = all_losses['obj_on_l1']
            loss_kl = all_losses['kl']
            loss_rec = all_losses['loss_rec']
            loss_kl_kp = all_losses['loss_kl_kp']
            loss_kl_feat = all_losses['loss_kl_feat']
            loss_kl_scale = all_losses['loss_kl_scale']
            loss_kl_depth = all_losses['loss_kl_depth']
            loss_kl_obj_on = all_losses['loss_kl_obj_on']

            if use_tracking:
                x = x.view(-1, *x.shape[2:])
                x_prior = x_prior.view(-1, *x_prior.shape[2:])
            # for plotting, confidence calculation
            mu_tot = z_base + mu_offset
            logvar_tot = logvar_offset
            # log
            batch_psnrs.append(psnr.data.cpu().item())
            batch_losses.append(loss.data.cpu().item())
            batch_losses_rec.append(loss_rec.data.cpu().item())
            batch_losses_kl.append(loss_kl.data.cpu().item())
            batch_losses_kl_kp.append(loss_kl_kp.data.cpu().item())
            batch_losses_kl_feat.append(loss_kl_feat.data.cpu().item())
            batch_losses_kl_scale.append(loss_kl_scale.data.cpu().item())
            batch_losses_kl_depth.append(loss_kl_depth.data.cpu().item())
            batch_losses_kl_obj_on.append(loss_kl_obj_on.data.cpu().item())
            # progress bar
            if epoch < warmup_epoch:
                pbar.set_description_str(f'epoch #{epoch} (warmup)')
            elif noisy:
                pbar.set_description_str(f'epoch #{epoch} (noisy)')
            else:
                pbar.set_description_str(f'epoch #{epoch}')
            pbar.set_postfix(loss=loss.data.cpu().item(), rec=loss_rec.data.cpu().item(),
                             kl=loss_kl.data.cpu().item(), on_l1=obj_on_l1.cpu().item())
            # break  # for debug
        pbar.close()
        losses.append(np.mean(batch_losses))
        losses_rec.append(np.mean(batch_losses_rec))
        losses_kl.append(np.mean(batch_losses_kl))
        losses_kl_kp.append(np.mean(batch_losses_kl_kp))
        losses_kl_feat.append(np.mean(batch_losses_kl_feat))
        losses_kl_scale.append(np.mean(batch_losses_kl_scale))
        losses_kl_depth.append(np.mean(batch_losses_kl_depth))
        losses_kl_obj_on.append(np.mean(batch_losses_kl_obj_on))
        if len(batch_psnrs) > 0:
            psnrs.append(np.mean(batch_psnrs))
        # scheduler
        scheduler.step()

        # epoch summary
        log_str = f'epoch {epoch} summary\n'
        log_str += f'loss: {losses[-1]:.3f}, rec: {losses_rec[-1]:.3f}, kl: {losses_kl[-1]:.3f}\n'
        log_str += f'kl_balance: {kl_balance:.3f}, kl_kp: {losses_kl_kp[-1]:.3f}, kl_feat: {losses_kl_feat[-1]:.3f}\n'
        log_str += f'kl_scale: {losses_kl_scale[-1]:.3f}, kl_depth: {losses_kl_depth[-1]:.3f}, kl_obj_on: {losses_kl_obj_on[-1]:.3f}\n'

        # log_str += f'mu max: {mu.max()}, mu min: {mu.min()}\n'
        log_str += f'mu max: {mu_tot.max()}, mu min: {mu_tot.min()}\n'
        log_str += f'mu offset max: {mu_offset.max()}, mu offset min: {mu_offset.min()}\n'
        log_str += f'val loss (freq: {eval_epoch_freq}): {valid_loss:.3f},' \
                   f' best: {best_valid_loss:.3f} @ epoch: {best_valid_epoch}\n'
        if obj_on is not None:
            log_str += f'obj_on max: {obj_on.max()}, obj_on min: {obj_on.min()}\n'
            log_str += f'scale max: {mu_scale.sigmoid().max()}, scale min: {mu_scale.sigmoid().min()}\n'
            log_str += f'depth max: {mu_depth.max()}, depth min: {mu_depth.min()}\n'
        if eval_im_metrics:
            log_str += f'val lpips (freq: {eval_epoch_freq}): {val_lpips:.3f},' \
                       f' best: {best_val_lpips:.3f} @ epoch: {best_val_lpips_epoch}\n'
        print(log_str)
        log_line(log_dir, log_str)

        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            # for plotting purposes
            mu_plot = mu_tot.clamp(min=kp_range[0], max=kp_range[1])
            max_imgs = 8
            img_with_kp = plot_keypoints_on_image_batch(mu_plot, x, radius=3,
                                                        thickness=1, max_imgs=max_imgs, kp_range=kp_range)
            img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x_prior, radius=3, thickness=1, max_imgs=max_imgs,
                                                          kp_range=kp_range)
            # top-k
            with torch.no_grad():
                logvar_sum = logvar_tot.sum(-1) * obj_on  # [bs, n_kp]
                logvar_topk = torch.topk(logvar_sum, k=topk, dim=-1, largest=False)
                indices = logvar_topk[1]  # [batch_size, topk]
                batch_indices = torch.arange(mu_tot.shape[0]).view(-1, 1).to(mu_tot.device)
                topk_kp = mu_tot[batch_indices, indices]
                # bounding boxes
                bb_scores = -1 * logvar_sum
                hard_threshold = None

            kp_batch = mu_plot
            scale_batch = mu_scale
            img_with_masks_nms, nms_ind = plot_bb_on_image_batch_from_z_scale_nms(kp_batch, scale_batch, x,
                                                                                  scores=bb_scores,
                                                                                  iou_thresh=iou_thresh,
                                                                                  thickness=1, max_imgs=max_imgs,
                                                                                  hard_thresh=hard_threshold)
            alpha_masks = torch.where(alpha_masks < 0.05, 0.0, 1.0)
            img_with_masks_alpha_nms, _ = plot_bb_on_image_batch_from_masks_nms(alpha_masks, x, scores=bb_scores,
                                                                                iou_thresh=iou_thresh, thickness=1,
                                                                                max_imgs=max_imgs,
                                                                                hard_thresh=hard_threshold)
            # hard_thresh: a general threshold for bb scores (set None to not use it)
            bb_str = f'bb scores: max: {bb_scores.max():.2f}, min: {bb_scores.min():.2f},' \
                     f' mean: {bb_scores.mean():.2f}\n'
            print(bb_str)
            log_line(log_dir, bb_str)
            img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                             radius=3, thickness=1, max_imgs=max_imgs,
                                                             kp_range=kp_range)
            dec_objects = model_output['dec_objects']
            bg = model_output['bg']
            vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(device),
                                         rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(device),
                                         img_with_kp_topk[:max_imgs, -3:].to(device),
                                         dec_objects[:max_imgs, -3:],
                                         img_with_masks_nms[:max_imgs, -3:].to(device),
                                         img_with_masks_alpha_nms[:max_imgs, -3:].to(device),
                                         bg[:max_imgs, -3:]],
                                        dim=0).data.cpu(), '{}/image_{}.jpg'.format(fig_dir, epoch),
                              nrow=8, pad_value=1)
            with torch.no_grad():
                _, dec_objects_rgb = torch.split(dec_objects_original, [1, 3], dim=2)
                dec_objects_rgb = dec_objects_rgb.reshape(-1, *dec_objects_rgb.shape[2:])
                cropped_objects_original = cropped_objects_original.clone().reshape(-1, 3,
                                                                                    cropped_objects_original.shape[
                                                                                        -1],
                                                                                    cropped_objects_original.shape[
                                                                                        -1])
                if cropped_objects_original.shape[-1] != dec_objects_rgb.shape[-1]:
                    cropped_objects_original = F.interpolate(cropped_objects_original,
                                                             size=dec_objects_rgb.shape[-1],
                                                             align_corners=False, mode='bilinear')
            vutils.save_image(
                torch.cat([cropped_objects_original[:max_imgs * 2, -3:], dec_objects_rgb[:max_imgs * 2, -3:]],
                          dim=0).data.cpu(), '{}/image_obj_{}.jpg'.format(fig_dir, epoch),
                nrow=8, pad_value=1)

            torch.save(model.state_dict(), os.path.join(save_dir, f'{ds}_dlp{run_prefix}.pth'))
            print("validation step...")
            valid_loss = evaluate_validation_elbo(model, config, epoch, batch_size=batch_size,
                                                  recon_loss_type=recon_loss_type, device=device,
                                                  save_image=True, fig_dir=fig_dir, topk=topk,
                                                  recon_loss_func=recon_loss_func, beta_rec=beta_rec,
                                                  iou_thresh=iou_thresh,
                                                  beta_kl=beta_kl, kl_balance=kl_balance)
            log_str = f'validation loss: {valid_loss:.3f}\n'
            print(log_str)
            log_line(log_dir, log_str)
            if best_valid_loss > valid_loss:
                log_str = f'validation loss updated: {best_valid_loss:.3f} -> {valid_loss:.3f}\n'
                print(log_str)
                log_line(log_dir, log_str)
                best_valid_loss = valid_loss
                best_valid_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(save_dir,
                                        f'{ds}_dlp{run_prefix}_best.pth'))
            torch.cuda.empty_cache()
            if eval_im_metrics and epoch > 0:
                valid_imm_results = eval_dlp_im_metric(model, device, config,
                                                       val_mode='val',
                                                       eval_dir=log_dir,
                                                       batch_size=batch_size)
                log_str = f'validation: lpips: {valid_imm_results["lpips"]:.3f}, '
                log_str += f'psnr: {valid_imm_results["psnr"]:.3f}, ssim: {valid_imm_results["ssim"]:.3f}\n'
                val_lpips = valid_imm_results['lpips']
                print(log_str)
                log_line(log_dir, log_str)
                if (not torch.isinf(torch.tensor(val_lpips))) and (best_val_lpips > val_lpips):
                    log_str = f'validation lpips updated: {best_val_lpips:.3f} -> {val_lpips:.3f}\n'
                    print(log_str)
                    log_line(log_dir, log_str)
                    best_val_lpips = val_lpips
                    best_val_lpips_epoch = epoch
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, f'{ds}_dlp{run_prefix}_best_lpips.pth'))
                torch.cuda.empty_cache()
        valid_losses.append(valid_loss)
        if eval_im_metrics:
            val_lpipss.append(val_lpips)
        # plot graphs
        if epoch > 0:
            num_plots = 4
            fig = plt.figure()
            ax = fig.add_subplot(num_plots, 1, 1)
            ax.plot(np.arange(len(losses[1:])), losses[1:], label="loss")
            ax.set_title(run_name)
            ax.legend()

            ax = fig.add_subplot(num_plots, 1, 2)
            ax.plot(np.arange(len(losses_kl[1:])), losses_kl[1:], label="kl", color='red')
            if learned_feature_dim > 0:
                ax.plot(np.arange(len(losses_kl_kp[1:])), losses_kl_kp[1:], label="kl_kp", color='cyan')
                ax.plot(np.arange(len(losses_kl_feat[1:])), losses_kl_feat[1:], label="kl_feat", color='green')
            ax.legend()

            ax = fig.add_subplot(num_plots, 1, 3)
            ax.plot(np.arange(len(losses_rec[1:])), losses_rec[1:], label="rec", color='green')
            ax.legend()

            ax = fig.add_subplot(num_plots, 1, 4)
            ax.plot(np.arange(len(valid_losses[1:])), valid_losses[1:], label="valid_loss", color='magenta')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{fig_dir}/{run_name}_graph.jpg')
            plt.close('all')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLP Single-GPU Training")
    parser.add_argument("-d", "--dataset", type=str, default='shapes',
                        help="dataset of to train the model on: ['traffic', 'clevrer', 'obj3d128', 'phyre']")
    args = parser.parse_args()
    ds = args.dataset
    if ds.endswith('json'):
        conf_path = ds
    else:
        conf_path = os.path.join('./configs', f'{ds}-img.json')

    train_dlp(conf_path)
