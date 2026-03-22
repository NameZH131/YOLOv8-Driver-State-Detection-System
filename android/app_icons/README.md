# App Icon Variants

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
