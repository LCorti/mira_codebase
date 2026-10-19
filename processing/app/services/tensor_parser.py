import base64
import io

import torch


def tensor_to_string(tensor) -> str | None:
    """Convert tensor to string representation for MongoDB storage."""
    if tensor is None:
        return None

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def string_to_tensor(tensor_string) -> torch.Tensor | None:
    """Convert string representation back to tensor."""
    if tensor_string is None:
        return torch.Tensor()  # Return empty tensor if None

    try:
        tensor_bytes = base64.b64decode(tensor_string.encode("utf-8"))
        buffer = io.BytesIO(tensor_bytes)
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Error converting string to tensor: {e}")
        return torch.Tensor()  # Return empty tensor on error
