"""Compatibility shims that let the ImageReward / BLIP reward model run on newer `transformers`.

ImageReward imports utilities that newer `transformers` releases moved or removed. The patch:
1. re-adds `apply_chunking_to_forward`, `find_pruneable_heads_and_indices` and `prune_linear_layer`;
2. mirrors public symbols between `transformers.modeling_utils` and `transformers.pytorch_utils`;
3. maps the legacy tokenizer attributes `additional_special_tokens(_ids)` to their new names;
4. restores `PreTrainedModel` helpers used by BLIP (`all_tied_weights_keys`, `get_head_mask`,
   `invert_attention_mask`, `get_extended_attention_mask`).
Each patch is applied only when the attribute is missing, so it is a no-op on older releases and
does not change reward values. It must run before `import ImageReward`; it also runs on import.
"""


def patch_transformers_for_imagereward():
    try:
        import torch
        import transformers.modeling_utils as mu
        import transformers.pytorch_utils as pu
    except ImportError:
        return

    # 1. Polyfills for functions removed from transformers >= 5.
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

    # 2. Mirror public symbols between modeling_utils and pytorch_utils.
    for attr in dir(pu):
        if not attr.startswith("_") and not hasattr(mu, attr):
            setattr(mu, attr, getattr(pu, attr))

    for attr in dir(mu):
        if not attr.startswith("_") and not hasattr(pu, attr):
            setattr(pu, attr, getattr(mu, attr))

    # 3. Resolve legacy tokenizer attributes.
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
        except ImportError:
            pass
    except ImportError:
        pass

    # 4. PreTrainedModel helpers removed in transformers >= 5.
    if not hasattr(mu.PreTrainedModel, "all_tied_weights_keys"):
        @property
        def all_tied_weights_keys(self):
            if not hasattr(self, "_all_tied_weights_keys_cached"):
                if hasattr(self, "get_expanded_tied_weights_keys"):
                    try:
                        self._all_tied_weights_keys_cached = self.get_expanded_tied_weights_keys(all_submodels=True)
                    except Exception:  # signature differs across releases; "no tied weights" is a safe default
                        self._all_tied_weights_keys_cached = {}
                elif hasattr(self, "_tied_weights_keys"):
                    self._all_tied_weights_keys_cached = dict(getattr(self, "_tied_weights_keys", {}))
                else:
                    self._all_tied_weights_keys_cached = {}
            return self._all_tied_weights_keys_cached

        @all_tied_weights_keys.setter
        def all_tied_weights_keys(self, val):
            self._all_tied_weights_keys_cached = val

        mu.PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys

    if not hasattr(mu.PreTrainedModel, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False):
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                dtype = getattr(self, "dtype", torch.float32)
                head_mask = head_mask.to(dtype=dtype)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        mu.PreTrainedModel.get_head_mask = get_head_mask
        if hasattr(mu, "ModuleUtilsMixin"):
            mu.ModuleUtilsMixin.get_head_mask = get_head_mask

    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            extended_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            extended_mask = encoder_attention_mask[:, None, None, :]
        else:
            extended_mask = encoder_attention_mask
        dtype = getattr(self, "dtype", torch.float32)
        extended_mask = extended_mask.to(dtype=dtype)
        try:
            min_val = torch.finfo(dtype).min
        except TypeError:
            min_val = -10000.0
        return (1.0 - extended_mask) * min_val

    if not hasattr(mu.PreTrainedModel, "invert_attention_mask"):
        mu.PreTrainedModel.invert_attention_mask = invert_attention_mask
    if hasattr(mu, "ModuleUtilsMixin") and not hasattr(mu.ModuleUtilsMixin, "invert_attention_mask"):
        mu.ModuleUtilsMixin.invert_attention_mask = invert_attention_mask

    try:
        from transformers.models.bert.modeling_bert import BertModel
        if not hasattr(BertModel, "invert_attention_mask"):
            BertModel.invert_attention_mask = invert_attention_mask
    except ImportError:
        pass

    try:
        import ImageReward.models.BLIP.med as blip_med
        if hasattr(blip_med, "BertModel") and not hasattr(blip_med.BertModel, "invert_attention_mask"):
            blip_med.BertModel.invert_attention_mask = invert_attention_mask
    except ImportError:
        pass

    def get_extended_attention_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
        if attention_mask.dim() == 3:
            extended_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_mask = attention_mask[:, None, None, :]
        else:
            extended_mask = attention_mask
        if dtype is None:
            dtype = getattr(self, "dtype", torch.float32)
        extended_mask = extended_mask.to(dtype=dtype)
        try:
            min_val = torch.finfo(dtype).min
        except TypeError:
            min_val = -10000.0
        return (1.0 - extended_mask) * min_val

    if not hasattr(mu.PreTrainedModel, "get_extended_attention_mask"):
        mu.PreTrainedModel.get_extended_attention_mask = get_extended_attention_mask
    if hasattr(mu, "ModuleUtilsMixin") and not hasattr(mu.ModuleUtilsMixin, "get_extended_attention_mask"):
        mu.ModuleUtilsMixin.get_extended_attention_mask = get_extended_attention_mask


patch_transformers_for_imagereward()
