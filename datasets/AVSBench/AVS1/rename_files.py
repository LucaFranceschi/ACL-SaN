import os

# apparently the same as find ms3/visual_frames -maxdepth 1 -type f -name "*.mp4*" -execdir bash -c 'mv "$0" "$(basename "$0" | sed "s/\.mp4//")"' {} \;
# not tried tho

if __name__ == '__main__':
    
    path = 'ms3/visual_frames'

    for filename in os.listdir(path):
        if '.mp4' in filename:
            
            new_filename = filename.replace('.mp4', '')
            
            try:
                # 4. Perform the rename operation
                os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
            except Exception as e:
                print(f"Could not rename '{filename}'. Error: {e}")