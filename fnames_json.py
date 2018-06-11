import os
import errno
import numpy as np
import random

def path_hierarchy(path):
    hierarchy = {
        'type': 'folder',
        'name': os.path.basename(path),
        'path': path,
        'id': random.sample(range(1,10000),1)[0]+300000,
    }
    try:
        hierarchy['children'] = [
            path_hierarchy(os.path.join(path, contents))
            for contents in os.listdir(path)
        ]
    except OSError as e:
        if e.errno != errno.ENOTDIR:
            raise
        hierarchy['type'] = 'file'

    return hierarchy

if __name__ == '__main__':
    import json
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="File structure JSON Generator")
    parser.add_argument('--dir', required = True, default = sys.argv[1] ,metavar = "/path/to/directory/to/parse", help = 'directory to convert to JSON')
    args = parser.parse_args()
    print("Converting Directory Structure to JSON Output:", args.dir)
    
    try:
        directory = args.dir
    except IndexError:
        directory = "."
    
    data = path_hierarchy(directory)
    
    with open('test_s2D.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)

    seg_file = json.load(open('test_s2D.json','r'))
    print("segfile_length:", len(seg_file["children"]))