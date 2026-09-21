from typing import Callable, Optional

from efficientvit.models.efficientvit import (
    EfficientViTSam,
    efficientvit_sam_l0,
    efficientvit_sam_l1,
    efficientvit_sam_l2,
    efficientvit_sam_xl0,
    efficientvit_sam_xl1,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file
from efficientvit.models.cais import CAISConfig, build_cais_efficientvit_sam
from efficientvit.models.vimr import VIMRConfig, VIMREfficientViTSam, build_vimr_efficientvit_sam

__all__ = [
    "create_efficientvit_sam_model",
    "REGISTERED_EFFICIENTVIT_SAM_MODEL",
    "REGISTERED_VIMR_MODEL",
    "REGISTERED_CAIS_MODEL",
]


REGISTERED_EFFICIENTVIT_SAM_MODEL: dict[str, tuple[Callable, float, str]] = {
    "efficientvit-sam-l0": (efficientvit_sam_l0, 1e-6, "assets/checkpoints/efficientvit_sam/efficientvit_sam_l0.pt"),
    "efficientvit-sam-l1": (efficientvit_sam_l1, 1e-6, "assets/checkpoints/efficientvit_sam/efficientvit_sam_l1.pt"),
    "efficientvit-sam-l2": (efficientvit_sam_l2, 1e-6, "assets/checkpoints/efficientvit_sam/efficientvit_sam_l2.pt"),
    "efficientvit-sam-xl0": (efficientvit_sam_xl0, 1e-6, "assets/checkpoints/efficientvit_sam/efficientvit_sam_xl0.pt"),
    "efficientvit-sam-xl1": (efficientvit_sam_xl1, 1e-6, "assets/checkpoints/efficientvit_sam/efficientvit_sam_xl1.pt"),
}

REGISTERED_VIMR_MODEL: dict[str, str] = {
    "efficientvit-sam-vimr": "efficientvit-sam-l0",
    "efficientvit-sam-vimr-l0": "efficientvit-sam-l0",
    "efficientvit-sam-vimr-l1": "efficientvit-sam-l1",
    "efficientvit-sam-vimr-l2": "efficientvit-sam-l2",
    "efficientvit-sam-vimr-xl0": "efficientvit-sam-xl0",
    "efficientvit-sam-vimr-xl1": "efficientvit-sam-xl1",
}

REGISTERED_CAIS_MODEL: dict[str, str] = {
    "efficientvit-sam-cais": "efficientvit-sam-l0",
    "efficientvit-sam-cais-l0": "efficientvit-sam-l0",
    "efficientvit-sam-cais-l1": "efficientvit-sam-l1",
    "efficientvit-sam-cais-l2": "efficientvit-sam-l2",
    "efficientvit-sam-cais-xl0": "efficientvit-sam-xl0",
    "efficientvit-sam-cais-xl1": "efficientvit-sam-xl1",
}


def create_efficientvit_sam_model(
    name: str, pretrained=True, weight_url: Optional[str] = None, **kwargs
) -> EfficientViTSam:
    if name in REGISTERED_CAIS_MODEL:
        base_name = REGISTERED_CAIS_MODEL[name]
        cais_config = CAISConfig.from_dict(kwargs.pop("cais_config", None))
        base_model = create_efficientvit_sam_model(base_name, pretrained=False, **kwargs)
        model = build_cais_efficientvit_sam(base_model, cais_config)
        _, norm_eps, default_pt = REGISTERED_EFFICIENTVIT_SAM_MODEL[base_name]
        set_norm_eps(model, norm_eps)
        weight_url = default_pt if weight_url is None else weight_url
        if pretrained:
            weight = load_state_dict_from_file(weight_url)
            incompatible = model.load_state_dict(weight, strict=False)
            loaded = len(set(weight).intersection(model.state_dict()))
            print(f"Loaded {loaded} pretrained EfficientViT-SAM parameters from {weight_url}")
            print(f"New CAIS parameters randomly initialized: {len(incompatible.missing_keys)} keys")
            if incompatible.unexpected_keys:
                print(f"Unexpected checkpoint keys: {len(incompatible.unexpected_keys)}")
        summary = model.parameter_summary()
        print(f"Total params: {summary['total']:,}; Trainable params: {summary['trainable']:,}")
        return model
    if name in REGISTERED_VIMR_MODEL:
        base_name = REGISTERED_VIMR_MODEL[name]
        vimr_config = VIMRConfig.from_dict(kwargs.pop("vimr_config", None))
        base_model = create_efficientvit_sam_model(base_name, pretrained=False, **kwargs)
        model = build_vimr_efficientvit_sam(base_model, vimr_config)
        _, norm_eps, default_pt = REGISTERED_EFFICIENTVIT_SAM_MODEL[base_name]
        set_norm_eps(model, norm_eps)
        weight_url = default_pt if weight_url is None else weight_url
        if pretrained:
            weight = load_state_dict_from_file(weight_url)
            incompatible = model.load_state_dict(weight, strict=False)
            loaded = len(set(weight).intersection(model.state_dict()))
            print(f"Loaded {loaded} pretrained EfficientViT-SAM parameters from {weight_url}")
            print(f"New VIMR parameters randomly initialized: {len(incompatible.missing_keys)} keys")
            if incompatible.unexpected_keys:
                print(f"Unexpected checkpoint keys: {len(incompatible.unexpected_keys)}")
        summary = model.parameter_summary()
        print(f"Total params: {summary['total']:,}; Trainable params: {summary['trainable']:,}")
        return model
    if name not in REGISTERED_EFFICIENTVIT_SAM_MODEL:
        raise ValueError(
            f"Cannot find {name} in the model zoo. List of models: "
            f"{list(REGISTERED_EFFICIENTVIT_SAM_MODEL) + list(REGISTERED_VIMR_MODEL) + list(REGISTERED_CAIS_MODEL)}"
        )
    else:
        model_cls, norm_eps, default_pt = REGISTERED_EFFICIENTVIT_SAM_MODEL[name]
        model = model_cls(**kwargs)
        set_norm_eps(model, norm_eps)
        weight_url = default_pt if weight_url is None else weight_url

    if pretrained:
        if weight_url is None:
            raise ValueError(f"Cannot find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model
