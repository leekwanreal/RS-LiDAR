"""
Compatibility patch for ImageReward with modern versions of Hugging Face transformers (>= 4.40).
ImageReward's BLIP/med.py imports:
  from transformers.modeling_utils import (
      PreTrainedModel,
      apply_chunking_to_forward,
      find_pruneable_heads_and_indices,
      prune_linear_layer,
  )
In transformers >= 4.40, apply_chunking_to_forward, find_pruneable_heads_and_indices,
and prune_linear_layer were relocated to transformers.pytorch_utils or deprecated.
This module guarantees these symbols exist on both transformers.modeling_utils
and transformers.pytorch_utils regardless of the installed transformers version.

Additionally, in modern transformers (>= 4.45 / 5.x), BertTokenizer removed the legacy
attribute `additional_special_tokens_ids`. ImageReward's BLIP/blip.py accesses
`tokenizer.additional_special_tokens_ids[0]`. This module dynamically provides
compatibility descriptors for `additional_special_tokens` and `additional_special_tokens_ids`.
"""

def patch_transformers_for_imagereward():
    try:
        import torch
        import transformers.modeling_utils
        import transformers.pytorch_utils
    except Exception:
        return

    # 1. apply_chunking_to_forward
    if not hasattr(transformers.modeling_utils, "apply_chunking_to_forward"):
        if hasattr(transformers.pytorch_utils, "apply_chunking_to_forward"):
            transformers.modeling_utils.apply_chunking_to_forward = transformers.pytorch_utils.apply_chunking_to_forward
        else:
            def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *args):
                assert len(args) > 0
                if chunk_size <= 0:
                    return forward_fn(*args)
                num_chunks = args[0].shape[chunk_dim] // chunk_size
                chunked_args = [torch.chunk(x, num_chunks, dim=chunk_dim) if isinstance(x, torch.Tensor) else [x] * num_chunks for x in args]
                layer_outputs = [forward_fn(*[x[i] for x in chunked_args]) for i in range(num_chunks)]
                return torch.cat(layer_outputs, dim=chunk_dim)
            transformers.modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward

    if hasattr(transformers, "pytorch_utils") and not hasattr(transformers.pytorch_utils, "apply_chunking_to_forward"):
        transformers.pytorch_utils.apply_chunking_to_forward = transformers.modeling_utils.apply_chunking_to_forward

    # 2. find_pruneable_heads_and_indices
    if not hasattr(transformers.modeling_utils, "find_pruneable_heads_and_indices"):
        if hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
            transformers.modeling_utils.find_pruneable_heads_and_indices = transformers.pytorch_utils.find_pruneable_heads_and_indices
        else:
            def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                if len(heads) == 0:
                    return set(), torch.empty(0, dtype=torch.long)
                heads = set(heads) - already_pruned_heads
                mask = torch.ones(n_heads, head_size)
                for head in heads:
                    head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                    mask[head] = 0
                mask = mask.view(-1).contiguous().eq(1)
                index = torch.arange(len(mask))[mask].long()
                return heads, index
            transformers.modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if hasattr(transformers, "pytorch_utils") and not hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
        transformers.pytorch_utils.find_pruneable_heads_and_indices = transformers.modeling_utils.find_pruneable_heads_and_indices

    # 3. prune_linear_layer
    if not hasattr(transformers.modeling_utils, "prune_linear_layer"):
        if hasattr(transformers.pytorch_utils, "prune_linear_layer"):
            transformers.modeling_utils.prune_linear_layer = transformers.pytorch_utils.prune_linear_layer
        else:
            def prune_linear_layer(layer, index, dim=0):
                import torch.nn as nn
                index = index.to(layer.weight.device)
                W = layer.weight.index_select(dim, index).clone().detach()
                if layer.bias is not None:
                    if dim == 1:
                        b = layer.bias.clone().detach()
                    else:
                        b = layer.bias[index].clone().detach()
                new_size = list(layer.weight.size())
                new_size[dim] = len(index)
                new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
                new_layer.weight.requires_grad = False
                new_layer.weight.copy_(W.contiguous())
                new_layer.weight.requires_grad = True
                if layer.bias is not None:
                    new_layer.bias.requires_grad = False
                    new_layer.bias.copy_(b.contiguous())
                    new_layer.bias.requires_grad = True
                return new_layer
            transformers.modeling_utils.prune_linear_layer = prune_linear_layer

    if hasattr(transformers, "pytorch_utils") and not hasattr(transformers.pytorch_utils, "prune_linear_layer"):
        transformers.pytorch_utils.prune_linear_layer = transformers.modeling_utils.prune_linear_layer

    # 4. Tokenizer additional_special_tokens & additional_special_tokens_ids
    try:
        from transformers import PreTrainedTokenizerBase

        def _get_additional_special_tokens_ids(self):
            if hasattr(self, "extra_special_tokens_ids"):
                return self.extra_special_tokens_ids
            if hasattr(self, "extra_special_tokens"):
                return [self.convert_tokens_to_ids(t) for t in self.extra_special_tokens]
            if hasattr(self, "_extra_special_tokens"):
                return [self.convert_tokens_to_ids(getattr(t, "content", str(t))) for t in self._extra_special_tokens]
            if hasattr(self, "additional_special_tokens"):
                return [self.convert_tokens_to_ids(t) for t in self.additional_special_tokens]
            return []

        def _get_additional_special_tokens(self):
            if hasattr(self, "extra_special_tokens"):
                return self.extra_special_tokens
            if hasattr(self, "_extra_special_tokens"):
                return [getattr(t, "content", str(t)) for t in self._extra_special_tokens]
            return []

        prop_ids = property(_get_additional_special_tokens_ids)
        prop_tokens = property(_get_additional_special_tokens)

        if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens_ids"):
            setattr(PreTrainedTokenizerBase, "additional_special_tokens_ids", prop_ids)
        if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
            setattr(PreTrainedTokenizerBase, "additional_special_tokens", prop_tokens)

        try:
            from transformers import BertTokenizer
            if not hasattr(BertTokenizer, "additional_special_tokens_ids"):
                setattr(BertTokenizer, "additional_special_tokens_ids", prop_ids)
            if not hasattr(BertTokenizer, "additional_special_tokens"):
                setattr(BertTokenizer, "additional_special_tokens", prop_tokens)
        except Exception:
            pass
    except Exception:
        pass

# Run patch upon module import
patch_transformers_for_imagereward()


