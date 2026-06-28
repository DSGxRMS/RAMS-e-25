from setuptools import setup

package_name = 'controls'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_data={
        # ship the Neural ODE checkpoint with the package so it is present in
        # the install tree (colcon would otherwise drop the .pt file).
        package_name: ['fs_model/*.pt'],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'control_node = controls.control_node:main',
        ],
    },
)
