from setuptools import setup

package_name = 'pathplanning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
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
            'pp_node_skidpad = pathplanning.pp_node_skidpad:main',
            'pp_tls = pathplanning.pp_tls:main',
            'pp_cone_memory = pathplanning.pp_cone_memory:main',
            'pp_hybrid_turn = pathplanning.pp_hybrid_turn:main',
            'pp_shadow_only = pathplanning.pp_shadow_only:main',
            'cone_memory_plot = pathplanning.cone_memory_plot:main',
            'pp_debug = pathplanning.pp_debug.skidpad_debug:main',
        ],
    },
)
