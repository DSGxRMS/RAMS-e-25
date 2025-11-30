from setuptools import setup
import os
from glob import glob
package_name = 'slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='armaanm',
    maintainer_email='armaanmahajanbg@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pred_node = slam.prediction_node:main',
            'slam_node = slam.slam_node:main',
            'slam_debug = slam.debug.slam_debug:main',
            'pred_debug = slam.debug.prediction_debug:main',
        ],
    },
)
