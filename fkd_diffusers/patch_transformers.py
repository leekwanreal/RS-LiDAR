"""
Universal compatibility bridge between modern Hugging Face transformers (>= 4.40 / 5.x)
and legacy multimodal / reward packages (such as ImageReward / BLIP).

Architectural Highlights (General Solution):
1. Dynamic Symbol Mirroring:
   Refactored utilities in Hugging Face (such as pruning, chunking, and tensor operations)
   were moved across `transformers.modeling_utils` and `transformers.pytorch_utils`.
   This module dynamically mirrors all public symbols bidirectionally between both packages.
2. Missing Symbol Polyfills:
   Mathematical fallback implementations for permanently deleted legacy functions
   (`find_pruneable_heads_and_indices`, `prune_linear_layer`, `apply_chunking_to_forward`).
3. Universal Tokenizer Backward-Compatibility Resolver:
   Intercepts `PreTrainedTokenizerBase.__getattr__` to dynamically map deprecated attributes
   (`additional_special_tokens_ids`, `additional_special_tokens`) to modern counterparts
   (`extra_special_tokens_ids`, `extra_special_tokens`).
"""

def patch_transformers_for_imagereward():
    try:
        import torch
        import transformers.modeling_utils as mu
        import transformers.pytorch_utils as pu
    except Exception:
        return

    # 1. Polyfills for functions deleted entirely from transformers >= 5
    if not hasattr(pu, "apply_chunking_to_forward") and not hasattr(mu, "apply_chunking_to_forward"):
        def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *args):
            assert len(args) > 0
            if chunk_size <= 0:
                return forward_fn(*args)
            num_chunks = args[0].shape[chunk_dim] // chunk_size
            chunked_args = [torch.chunk(x, num_chunks, dim=chunk_dim) if isinstance(x, torch.Tensor) else [x] * num_chunks for x in args]
            layer_outputs = [forward_fn(*[x[i] for x in chunked_args]) for i in range(num_chunks)]
            return torch.cat(layer_outputs, dim=chunk_dim)
        pu.apply_chunking_to_forward = apply_chunking_to_forward

    if not hasattr(pu, "find_pruneable_heads_and_indices") and not hasattr(mu, "find_pruneable_heads_and_indices"):
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
        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(pu, "prune_linear_layer") and not hasattr(mu, "prune_linear_layer"):
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
        pu.prune_linear_layer = prune_linear_layer

    # 2. Universal Dynamic Symbol Mirroring between modeling_utils and pytorch_utils
    for attr in dir(pu):
        if not attr.startswith("_") and not hasattr(mu, attr):
            try:
                setattr(mu, attr, getattr(pu, attr))
            except Exception:
                pass

    for attr in dir(mu):
        if not attr.startswith("_") and not hasattr(pu, attr):
            try:
                setattr(pu, attr, getattr(mu, attr))
            except Exception:
                pass

    # 3. Universal Tokenizer Compatibility (Dynamic Attribute Resolution)
    try:
        from transformers import PreTrainedTokenizerBase

        LEGACY_TOKENIZER_RESOLVER = {
            "additional_special_tokens_ids": lambda tok: (
                getattr(tok, "extra_special_tokens_ids", None)
                or [tok.convert_tokens_to_ids(getattr(t, "content", str(t))) for t in getattr(tok, "extra_special_tokens", getattr(tok, "_extra_special_tokens", []))]
            ),
            "additional_special_tokens": lambda tok: (
                getattr(tok, "extra_special_tokens", None)
                or [getattr(t, "content", str(t)) for t in getattr(tok, "_extra_special_tokens", [])]
            ),
        }

        orig_getattr = getattr(PreTrainedTokenizerBase, "__getattr__", None)
        if orig_getattr is not None:
            def universal_tokenizer_getattr(self, key):
                if key in LEGACY_TOKENIZER_RESOLVER:
                    return LEGACY_TOKENIZER_RESOLVER[key](self)
                return orig_getattr(self, key)
            PreTrainedTokenizerBase.__getattr__ = universal_tokenizer_getattr

        prop_ids = property(LEGACY_TOKENIZER_RESOLVER["additional_special_tokens_ids"])
        prop_tokens = property(LEGACY_TOKENIZER_RESOLVER["additional_special_tokens"])
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



