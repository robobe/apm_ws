from setuptools import setup
import os

package_name = 'apm_demos'

data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]

def package_files(data_files, directory_list):
    paths_dict = {}
    for directory in directory_list:
        for (path, directories, filenames) in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', package_name, path)
                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)
                else:
                    paths_dict[install_path] = [file_path]

    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))

    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=package_files(data_files, [
        'launch/',
    ]),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='robo2020@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "test=apm_demos.simple_arm_and_takeoff:main",
            "simulate_optitrack=apm_demos.simulate_optitrack:main",
            "pose_republisher=apm_demos.pose_republisher:main",
            "set_origin=apm_demos.set_origin:main",
            "mav_reader_demo=apm_demos.mav_reader_demo:main",
            "mav_writer_demo=apm_demos.mav_writer_demo:main",
            "home_ekf=apm_demos.home_and_ekf_demo:main",
            "simple_server=apm_demos.simple_service_demo:main",
            "simple_client=apm_demos.simple_client_demo:main",
            "rangefinder=apm_demos.rangefinder_demo:main",
            "param_demo=apm_demos.params_demo:main"
        ],
    },
)
