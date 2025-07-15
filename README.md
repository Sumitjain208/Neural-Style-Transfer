# Neural Style Transfer

This project implements Neural Style Transfer using PyTorch, allowing you to combine the content of one image with the artistic style of another image.

## Features

- Implementation of Neural Style Transfer using VGG19
- Support for custom content and style images
- Adjustable content and style weights
- Progress tracking during style transfer
- Visualization of results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your content and style images in the project directory
2. Run the script:
```bash
python neural_style_transfer.py
```

3. The stylized image will be saved as 'output.jpg' and displayed using matplotlib

## Parameters

You can adjust the following parameters in the script:
- `num_steps`: Number of optimization steps (default: 300)
- `content_weight`: Weight of content loss (default: 1)
- `style_weight`: Weight of style loss (default: 1e6)
- `max_size`: Maximum size of the input images (default: 400)

## Example

```python
result = style_transfer(
    content_path="content.jpg",
    style_path="style.jpeg",
    output_path="output.jpg",
    num_steps=300,
    content_weight=1,
    style_weight=1e6
)
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- matplotlib
- numpy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the original Neural Style Transfer paper:
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Gatys et al.
