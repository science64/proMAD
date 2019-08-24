from setuptools import setup
from pathlib import Path
from proMAD import config

base_dir = Path(__file__).absolute().parent
read_me = base_dir / 'README.md'
long_description = read_me.read_text(encoding='utf-8')
version = config.version

setup(name=config.app_name,
      version=version,
      description='Semiquantitative densitometric measurement of protein microarrays',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://proMAD.dev',
      download_url=f'https://github.com/theia-dev/proMAD/archive/v{version}.zip',
      author=config.app_author,
      author_email='',
      license='MIT',
      packages=['proMAD'],
      include_package_data=True,
      install_requires=['numpy', 'matplotlib', 'scipy', 'scikit-image', 'imageio', 'openpyxl', 'requests', 'ipython', 'xmltodict'],
      zip_safe=True,
      keywords=['protein', 'microarrays', 'densitometric'],
      python_requires='~=3.6',
      classifiers=[
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'Topic :: Scientific/Engineering :: Information Analysis',

          'License :: OSI Approved :: MIT License',

          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3 :: Only'
      ],
      )
