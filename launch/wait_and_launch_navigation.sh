#!/bin/bash

echo "Waiting for /odom topic..."
rostopic echo -n 1 /odom > /dev/null # Wait for one message on /odom
if [ $? -ne 0 ]; then
  echo "Failed to receive /odom topic. Exiting."
  exit 1
fi
echo "/odom topic received. Launching navigation."

# Now, launch your navigation node
roslaunch jupiterobot2_navigation jupiterobot2_navigation.launch map_file:=/home/mustar/catkin_ws/maps/robotlab_naomi.yaml
# roslaunch robotedge_jupiter_navigation jh_test.launch

# roslaunch jupiterobot2_bringup arm_head_bringup.launch