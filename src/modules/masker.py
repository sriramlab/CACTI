import torch
import warnings

class ReMaskerModule:
    @staticmethod
    def random_masking(x, m, mask_ratio, shuffle=False, eps=1e-6):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        m : [N,L] ; 1: obs, 0: miss
        """
        N, L, D = x.shape  # batch, length, dim
        if shuffle:
            len_keep = int(L * (1 - mask_ratio))
        else:
            len_keep = int(torch.min(torch.sum(m, dim=1)))
            # print(f"eval len_keep: {len_keep}")

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        noise[m < eps] = 1
        num_masked = (noise == 1).sum(dim=1)

        if num_masked.max() > L - len_keep:
            warnings.warn(f"max mased: {num_masked.max()}, len_keep: {len_keep}")
            len_keep = L - num_masked.max()
        # print(f"random_masking len_keep: {len_keep}")

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        nask = torch.ones([N, L], device=x.device) - mask

        if shuffle:
            mask[m < eps] = 0

        return x_masked, mask, nask, ids_restore

    @staticmethod
    def unshuffle(x, ids_restore, mask_token):
        # append mask tokens to sequence
        mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        return x_

# To be used for copy-masks
class CopySegmentModule:
    @staticmethod
    def contextify(x, mask, pad, shuffle=False, threshold=0.99):
        B, L, D = x.shape  # batch, length, dim

        # Calculate number of unmasked tokens per sample
        num_obs = (1-mask).sum(dim=1) # mask 1: miss, 0, obs

        # Shuffle indices if training/requested
        if shuffle:
            # get median obs len of batch
            obs_len = int(torch.median(num_obs.type(torch.float32))) # round down
            # get rand noise
            noise = torch.rand(B, L, device=x.device)
        else:
            # get min observed features
            # NOTE: this assumses during inference, there's only 1 sample per batch
            obs_len = int(torch.min(num_obs))
            # ordered noise
            interval = threshold/L
            noise = torch.arange(0, threshold, interval, device=x.device).unsqueeze(0).expand(B, -1)

        # print(f"Contextify len_keep: {obs_len}, obs_max: {int(torch.max(num_obs))}, int_mean: {int(torch.mean(num_obs.type(torch.float32)))}, int_median: {int(torch.median(num_obs.type(torch.float32)))}")
            
        # cast all (miss U masked) to = 1
        noise[mask > threshold] = 1.0 
        # shuffle map
        ids_shuffle = torch.argsort(noise, dim=1) # small: keep, large: drop

        # restore map for decoder
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep obs_len subset for encoder
        ids_keep = ids_shuffle[:, :obs_len]

        x_masked = x.clone()

        x_masked[mask > threshold] = pad 

        x_masked = torch.gather(x_masked, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # observed mask  1 is keep, 0 is remove
        # omask = torch.zeros([B, L], device=x.device)
        omask = torch.ones([B, L], device=x.device)

        omask[:, obs_len:] = 0
        omask = torch.gather(omask, dim=1, index=ids_restore)
        omask *= 1-mask

        return x_masked, ids_restore, omask

    @staticmethod
    def decontextify(x_processed, ids_restore, omask, token):
        B, L = ids_restore.shape  # batch, length

        _, obs_len, D = x_processed.shape # batch, obs length, dim

        # token_pad : shape (1,1,D)
        # make full size tensor
        x_shuffled = torch.cat([x_processed, token.expand(B, L - obs_len, -1)], dim=1)
        
        # unshuffle
        x_unshuffled = torch.gather(x_shuffled, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # cast all un-obsrved to mask token
        x_unshuffled[omask < 1] = token

        return x_unshuffled

class MaskModule:
    @staticmethod
    def contextify(x, mask, shuffle=False, threshold=0.99):
        B, L, D = x.shape  # batch, length, dim

        # Calculate number of unmasked tokens per sample
        num_obs = (1-mask).sum(dim=1) # mask 1: miss, 0, obs

        # get min observed features
        obs_len = int(torch.min(num_obs))

        # print(f"Contextify len_keep: {obs_len}, obs_max: {int(torch.max(num_obs))}, int_mean: {int(torch.mean(num_obs.type(torch.float32)))}, int_median: {int(torch.median(num_obs.type(torch.float32)))}")

        # Shuffle indices if requested
        if shuffle:
            # get rand noise
            noise = torch.rand(B, L, device=x.device)
        else:
            # ordered noise
            interval = threshold/L
            noise = torch.arange(0, threshold, interval, device=x.device).unsqueeze(0).expand(B, -1)
            
        # cast all (miss U masked) to = 1
        noise[mask > threshold] = 1 
        # shuffle map
        ids_shuffle = torch.argsort(noise, dim=1) # small: keep, large: drop

        # restore map for decoder
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep obs_len subset for encoder
        ids_keep = ids_shuffle[:, :obs_len]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # observed mask  1 is keep, 0 is remove
        omask = torch.zeros([B, L], device=x.device)
        omask[:, :obs_len] = 1
        omask = torch.gather(omask, dim=1, index=ids_restore)

        return x_masked, ids_restore, omask

    @staticmethod
    def decontextify(x, ids_restore, token):
        B, L = ids_restore.shape  # batch, length

        _, obs_len, D = x.shape # batch, obs length, dim

        token_pad = token.repeat(B, L - obs_len, 1)

        # make full size tensor
        x_shuffled = torch.cat([x, token_pad], dim=1)
        
        # unshuffle
        x_unshuffled = torch.gather(x_shuffled, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  

        return x_unshuffled
