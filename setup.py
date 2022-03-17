from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent

setup(name='keras-pruning',
      version='0.9.3',
      description='PyTorch training manager (Public Beta 4)',
      author='Kison Ho',
      author_email='unfit-gothic.0q@icloud.com',
      packages=['keras_pruning',
            'keras_pruning.core',
            'keras_pruning.metrics',
            'keras_pruning.sparsity',
            'keras_pruning.train'],
      package_dir={
            'keras_pruning': 'lib',
            'keras_pruning.core': 'lib/core',
            'keras_pruning.metrics': 'lib/metrics',
            'keras_pruning.sparsity': 'lib/sparsity',
            'keras_pruning.train': 'lib/train'
      },
      python_requires=">=3.9",
      requires=["tensorflow"],
      url="https://github.com/kisonho/keras-pruning.git"
)
