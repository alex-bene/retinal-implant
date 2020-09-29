from distutils.core import setup

setup(
  name = 'retinal_implant_utils',
  packages = ['retinal_implant_utils'],
  version = '0.0.1',
  license='MIT',
  description = 'Utilities to help with retinal implants simulation and evaluation,
  author = 'Alexandros Benetatos',
  author_email = 'alexandrosbene@gmail.com',
  url = 'https://github.com/alex-bene/retinal-implants',
  download_url = 'https://github.com/alex-bene/retinal-implants/archive/v0.0.1-beta.tar.gz',
  keywords = ['retinal implants', 'pulse2percept', 'utilities'],
  install_requires=[
          'tqdm',
          'numpy',
          'torch',
          'pillow',
          'pickle',
          'skimage',
          'pytorchUtils',
          'pulse2percept',
                   ],
  classifiers=[
    'Development Status :: 4 - Beta', # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Operating System :: OS Independent',
              ],
)