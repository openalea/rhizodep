import imageio
from path import Path

filenames = Path('video').glob('*.png')
filenames = sorted(filenames)
with imageio.get_writer('root_movie.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(str(filename))
        writer.append_data(image)