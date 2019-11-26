# 6-881-examples

This repository contains a collection of tools for interacting with robots
 and cameras, developed to support the Intelligent Robot Manipulation class [(MIT 6.881)](http://manipulation.csail.mit.edu/).

## Docker Use with Graphics Support

In the root directory of this repository, run 
```bash
$ python3 docker_run.py --os [your operating system] -t drake-20191125
``` 
where `[your operating system]` should be replaced with `mac` or `linux
`. This command will start a docker container (virtual machine) with the
 docker image you have created. The `6-881-examples` folder on the host
  machine (your laptop/desktop) is mounted to `/psets` in the docker
   container. If your MacOS version is not Catalina, you may need to
    use an earlier version of the `docker_run.py` script which can be found
     [here](http://manipulation.csail.mit.edu/install_drake_docker.html).

In the docker container, run the following commands to start the jupyter
 notebook server with graphics support:
```bash
$ cd /psets
$ Xvfb :100 -ac -screen 0 800x600x24 &
$ DISPLAY=:100 jupyter notebook --ip 0.0.0.0 --port 8080 --allow-root --no-browser
```

## Contents
- `robot_control` contains the part of problem set 5 which runs
 three different controllers on a simulated robot. 
- `camera_sim` includes examples of using Drake's RGBD camera simulators
. Specifically, `rgbd_camera_simulation.ipynb` shows how to construct from
 scratch a `Diagram` system that includes a RGBD camera simulator; `run_cameras_manipulation_station.ipynb` shows how to access the three
  RGBD cameras already inside `ManipulationStation`.

