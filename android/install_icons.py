#!/usr/bin/env python3
"""
Install generated icons to Android project
"""
import os
import shutil

def install_icons(quality='high'):
    """Install icons of specified quality to Android resources"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, 'app_icons', quality)
    res_dir = os.path.join(script_dir, 'app', 'src', 'main', 'res')
    
    print(f"\nInstalling {quality.upper()} quality icons...")
    print(f"Source: {source_dir}")
    print(f"Target: {res_dir}")
    
    # Mipmap density mappings
    mipmap_dirs = {
        'mdpi': 'mipmap-mdpi',
        'hdpi': 'mipmap-hdpi',
        'xhdpi': 'mipmap-xhdpi',
        'xxhdpi': 'mipmap-xxhdpi',
        'xxxhdpi': 'mipmap-xxxhdpi',
    }
    
    # Copy mipmap icons
    for density, dirname in mipmap_dirs.items():
        source_file = os.path.join(source_dir, f'ic_launcher_{density}.png')
        target_dir = os.path.join(res_dir, dirname)
        target_file = os.path.join(target_dir, 'ic_launcher.png')
        
        os.makedirs(target_dir, exist_ok=True)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            print(f"  OK {dirname}/ic_launcher.png")
        else:
            print(f"  X Missing: {source_file}")
    
    # Copy adaptive icon foreground
    foreground_source = os.path.join(source_dir, 'ic_launcher_foreground.png')
    foreground_target = os.path.join(res_dir, 'drawable', 'ic_launcher_foreground.png')
    
    os.makedirs(os.path.dirname(foreground_target), exist_ok=True)
    if os.path.exists(foreground_source):
        shutil.copy2(foreground_source, foreground_target)
        print(f"  OK drawable/ic_launcher_foreground.png")
    
    print(f"\n[OK] Icons installed successfully!")
    print(f"\nTo use the new icons:")
    print(f"  1. Open Android Studio")
    print(f"  2. Clean and rebuild the project (Build > Clean Project)")
    print(f"  3. The new icons will appear in the app")

if __name__ == '__main__':
    import sys
    
    quality = sys.argv[1] if len(sys.argv) > 1 else 'high'
    
    if quality not in ['low', 'medium', 'high']:
        print("Usage: python install_icons.py [low|medium|high]")
        print("  default: high")
        exit(1)
    
    install_icons(quality)
