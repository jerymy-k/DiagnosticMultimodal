from pathlib import Path


def parse_line(line: str, num_classes: int):
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
    except ValueError:
        return None
    if not 0 <= cls < num_classes:
        return None
    if not all(0 <= v <= 1 for v in (xc, yc, w, h)) or w <= 0 or h <= 0:
        return None
    return cls, xc, yc, w, h


def read_labels(label_path: Path, num_classes: int):
    boxes, invalid = [], []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        box = parse_line(line, num_classes)
        if box is None:
            invalid.append(line)
        else:
            boxes.append(box)
    return boxes, invalid
