#!/usr/bin/env python3
"""
Android App Icon Generator
Generates launcher icons in all mipmap densities from source image
"""
import os
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np

# Icon dimensions for each mipmap density
MIPMAP_SIZES = {
    'mdpi': 48,
    'hdpi': 72,
    'xhdpi': 96,
    'xxhdpi': 144,
    'xxxhdpi': 192,
}

# Quality presets
QUALITY_PRESETS = {
    'high': {
        'resize_quality': Image.LANCZOS,
        'sharpen': True,
        'enhance_colors': True,
        'edge_enhance': True,
    },
    'medium': {
        'resize_quality': Image.BILINEAR,
        'sharpen': False,
        'enhance_colors': True,
        'edge_enhance': False,
    },
    'low': {
        'resize_quality': Image.BOX,
        'sharpen': False,
        'enhance_colors': False,
        'edge_enhance': False,
    }
}

def load_source_image(path):
    """Load and validate source image"""
    img = Image.open(path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    return img

def create_circular_mask(size):
    """Create a circular mask for the icon"""
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    margin = size // 10
    draw.ellipse([margin, margin, size - margin, size - margin], fill=255)
    return mask

def process_image(img, size, quality_preset):
    """Process image to create icon at specified size"""
    settings = QUALITY_PRESETS[quality_preset]
    
    # Create a square canvas with padding for adaptive icon safe zone
    canvas_size = int(size * 1.2)
    canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
    
    # Calculate crop and resize
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    
    # Crop to square
    cropped = img.crop((left, top, left + min_dim, top + min_dim))
    
    # Resize
    resized = cropped.resize((canvas_size, canvas_size), settings['resize_quality'])
    
    # Apply enhancements for high quality
    if settings['enhance_colors']:
        enhancer = ImageEnhance.Color(resized)
        resized = enhancer.enhance(1.1)
    
    if settings['sharpen']:
        resized = resized.filter(ImageFilter.SHARPEN)
        resized = resized.filter(ImageFilter.EDGE_ENHANCE)
    
    # Create circular mask
    mask = create_circular_mask(canvas_size)
    
    # Apply mask
    output = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
    output.paste(resized, (0, 0), mask)
    
    return output

def create_adaptive_icon(img, quality_preset):
    """Create adaptive icon foreground (108x108 with safe zone)"""
    settings = QUALITY_PRESETS[quality_preset]
    size = 108
    
    # Create canvas with padding (safe zone)
    canvas_size = size
    canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
    
    # Calculate crop - use center of image
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    cropped = img.crop((left, top, left + min_dim, top + min_dim))
    
    # Resize to fit in safe zone (inner 66% of icon)
    safe_size = int(size * 0.66)
    offset = (size - safe_size) // 2
    
    resized = cropped.resize((safe_size, safe_size), settings['resize_quality'])
    
    if settings['enhance_colors']:
        enhancer = ImageEnhance.Color(resized)
        resized = enhancer.enhance(1.15)
    
    if settings['sharpen']:
        resized = resized.filter(ImageFilter.SHARPEN)
    
    # Paste centered
    canvas.paste(resized, (offset, offset))
    
    return canvas

def generate_icons(source_path, output_dir, quality='high'):
    """Generate all icon variants for a quality preset"""
    print(f"\n{'='*50}")
    print(f"Generating {quality.upper()} quality icons")
    print(f"{'='*50}")
    
    # Load source
    img = load_source_image(source_path)
    print(f"Source: {img.size[0]}x{img.size[1]}")
    
    # Create output directory for this quality
    quality_dir = os.path.join(output_dir, quality)
    os.makedirs(quality_dir, exist_ok=True)
    
    # Generate mipmap icons
    for density, size in MIPMAP_SIZES.items():
        icon = process_image(img, size, quality)
        icon_path = os.path.join(quality_dir, f'ic_launcher_{density}.png')
        icon.save(icon_path, 'PNG', optimize=True)
        print(f"  {density}: {icon_path}")
    
    # Generate adaptive icon foreground
    adaptive_icon = create_adaptive_icon(img, quality)
    adaptive_path = os.path.join(quality_dir, 'ic_launcher_foreground.png')
    adaptive_icon.save(adaptive_path, 'PNG', optimize=True)
    print(f"  adaptive: {adaptive_path}")
    
    print(f"\n{quality.upper()} icons generated successfully!")

def generate_all_qualities(source_path, output_dir):
    """Generate icons for all quality presets"""
    # Clean output directory
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate for each quality
    for quality in ['low', 'medium', 'high']:
        generate_icons(source_path, output_dir, quality)
    
    # Create a README
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("""# App Icon Variants

This directory contains app icons in three quality levels:

## Quality Levels

### High Quality (`high/`)
- Full resolution processing
- Color enhancement
- Sharpening applied
- Best for: Primary app store listings

### Medium Quality (`medium/`)
- Balanced processing
- Color enhancement only
- Best for: Development/testing

### Low Quality (`low/`)
- Minimal processing
- No enhancements
- Best for: Placeholder/backup

## Usage

Copy the desired quality folder contents to your Android project's `res/mipmap-*` directories:
- `mipmap-mdpi/` - 48x48
- `mipmap-hdpi/` - 72x72  
- `mipmap-xhdpi/` - 96x96
- `mipmap-xxhdpi/` - 144x144
- `mipmap-xxxhdpi/` - 192x192

For adaptive icons, update:
- `res/drawable/ic_launcher_foreground.png` (108x108)
- `res/drawable/ic_launcher_background.xml` (set your background color)
""")
    
    print(f"\n{'='*50}")
    print("All icon generation complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*50}")

if __name__ == '__main__':
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(script_dir, '屏幕截图 2026-03-22 232212.png')
    output_dir = os.path.join(script_dir, 'app_icons')
    
    # Check source exists
    if not os.path.exists(source_path):
        print(f"Error: Source image not found: {source_path}")
        exit(1)
    
    # Generate all icons
    generate_all_qualities(source_path, output_dir)
