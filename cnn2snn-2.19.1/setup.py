from os import path
from setuptools import setup
from platform import platform


def get_tf_dep():
  platform_string = platform()
  if 'macOS' in platform_string and 'arm64' in platform_string:
    tf_name = 'tensorflow-macos'
  else:
    tf_name = 'tensorflow'
  return tf_name + '~=2.19.0'


# Read the contents of the README file
directory = path.abspath(path.dirname(__file__))
with open(path.join(directory, 'README'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cnn2snn',
    version='2.19.1',
    description='Keras to Akida CNN Converter',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Johan Mejia',
    author_email='jmejia@brainchip.com',
    url='https://doc.brainchipinc.com',
    license='Apache 2.0',
    packages=['cnn2snn', 'cnn2snn.transforms', 'cnn2snn.calibration', 'cnn2snn.quantizeml',
              'cnn2snn.quantizeml.onnx_conversion'],
    entry_points={
        'console_scripts': [ 'cnn2snn = cnn2snn.cli:main' ]
    },
    install_requires=['tf_keras~=2.19.0', get_tf_dep(),
        'akida==2.19.1', 'quantizeml~=1.2.2'],
    python_requires='>=3.10',
)
