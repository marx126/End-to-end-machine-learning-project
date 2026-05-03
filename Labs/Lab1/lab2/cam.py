import matplotlib.pyplot as plt
import torch
from torch.nn.functional import softmax
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.io import read_image
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).eval()
preprocess = weights.transforms()
labels = weights.meta["categories"]

def class_id(name):
    """Return ImageNet class id from class name"""
    name = name.lower().replace(" ", "_")
    for i, label in enumerate(labels):
        if label.lower().replace(" ", "_") == name:
            return i
    raise ValueError(f"Class not found: {name}")

def predict(image_path, top_k=5):
    """Load image and print top predictions."""
    image = read_image(image_path)
    x = preprocess(image).unsqueeze(0)

    logits = model(x)
    probs = softmax(logits, dim=1)
    values, ids = torch.topk(probs, top_k)

    print("Top predictions:")
    for i, (prob, idx) in enumerate(zip(values[0], ids[0]), start=1):
        idx = idx.item()
        print(f"{i}. {labels[idx]} | id={idx} | prob={prob.item():.4f} | logit={logits[0, idx].item():.3f}")

    return image, x, logits, probs

def show_cam(image_path, target_class, layer=model.layer4):
    """Show the CAM for a given image and target class."""
    image = read_image(image_path)
    x = preprocess(image).unsqueeze(0)
    target_id = class_id(target_class) if isinstance(target_class, str) else target_class

    with SmoothGradCAMpp(model, target_layer=layer) as cam:
        logits = model(x)
        probs = softmax(logits, dim=1)
        activation_map = cam(target_id, logits)[0].squeeze(0)

    result = overlay_mask(
        to_pil_image(image),
        to_pil_image(activation_map, mode="F"),
        alpha=0.55,
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(result)
    plt.axis("off")
    plt.title(
        f"Target: {labels[target_id]}\n"
        f"prob={probs[0, target_id].item():.4f}, logit={logits[0, target_id].item():.3f}"
    )
    plt.show()

def compare_layers(image_path, target_class):
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for layer in layers:
        show_cam(image_path, target_class, layer=layer)