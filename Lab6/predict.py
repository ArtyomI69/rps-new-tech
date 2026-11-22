# predict.py
import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from train_zebra_horse import SimpleCNN, IDX_TO_CLASS


def load_model(ckpt_path, device):
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict(image_path, model, device, tf):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)          # (1, 3, H, W)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax().item())
    return {
        "path": image_path,
        "predicted_class": IDX_TO_CLASS[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "probs": {IDX_TO_CLASS[i]: float(probs[i].item()) for i in range(2)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="путь к одному .jpg/.png")
    ap.add_argument("--ckpt", default="outputs/best_model.pt")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f"Файл не найден: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    tf = build_transform(args.img_size)

    r = predict(args.image, model, device, tf)
    print(f"Файл:         {r['path']}")
    print(f"Класс:        {r['predicted_class']}")
    print(f"Уверенность:  {r['confidence']:.4f}")
    print(f"Вероятности:  "
          f"bay_horse={r['probs']['bay_horse']:.4f}, "
          f"zebra={r['probs']['zebra']:.4f}")


if __name__ == "__main__":
    main()