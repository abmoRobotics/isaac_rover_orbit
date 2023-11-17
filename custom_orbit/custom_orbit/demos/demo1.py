import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import torch
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.config.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG
from omni.isaac.orbit.robots.mobile_manipulator import MobileManipulator
from custom_orbit.robots.mobile_robot import MobileRobot
from custom_orbit.robots.config.aau_rover import AAU_ROVER_CFG
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

"""
Helpers
"""


def design_scene():
    """Add prims to the scene."""
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane")
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )


def main():
    """Spawns a mobile manipulator and applies random joint position commands."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Set main camera
    set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.0])
    # Spawn things into stage
    #robot = MobileManipulator(cfg=RIDGEBACK_FRANKA_PANDA_CFG)
    robot = MobileRobot(cfg=AAU_ROVER_CFG)
    num_envs = 2
    for i in range(num_envs):
        robot.spawn(f"/World/Robot_{i}", translation=(0.0, 2.0*i, 0.0))
        robot.prepare_contact_reporter("/World/defaultGroundPlane/GroundPlane/CollisionPlane")

    #robot.spawn(["/World/Robot_0", "/World/Robot_1"])
    #robot.prepare_contact_reporter("/World/defaultGroundPlane/GroundPlane/CollisionPlane")
    # robot.spawn("/World/Robot_1", translation=(0.0, -1.0, 0.0))
    # robot.spawn("/World/Robot_2", translation=(0.0, 1.0, 0.0))
    design_scene()
    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot.initialize("/World/Robot.*")
    # Reset states
    robot.reset_buffers()
    from omni.isaac.core.prims import RigidPrimView

    collisions_view = RigidPrimView("/World/Robot.*/.*",name="knees_view", reset_xform_properties=False, track_contact_forces=True, prepare_contact_sensors=False, 
                                   contact_filter_prim_paths_expr=["/World/defaultGroundPlane/GroundPlane/CollisionPlane"])
    
    #collisions_view = RigidPrimView("/World/Robot.*/.*rive",name="knees_view", reset_xform_properties=False, track_contact_forces=True, prepare_contact_sensors=False)
    collisions_view.initialize()
    #print(collisions_view)
    # from omni.physx import get_physx_simulation_interface
    # _contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact_report_event)
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy action
    # actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
    # actions[:, 0 : robot.base_num_dof] = 0.0
    # actions[:, -1] = -1

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # reset
        if ep_step_count % 100 == 0:
            print("NEW")
            #print(collisions_view.get_contact_force_data)
            #print(collisions_view.get_net_contact_forces().view(num_envs, 6, 3))
            print(collisions_view.get_contact_force_matrix().view(num_envs, -1, 3))
            print(collisions_view.get_contact_force_matrix().view(num_envs, -1, 3).shape)
            sim_time = 0.0
            ep_step_count = 0
            # reset dof state
            dof_pos, dof_vel = robot.get_default_dof_state()
            robot.set_dof_state(dof_pos, dof_vel)
            robot.reset_buffers()
            # reset command
            #actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
            #actions[:, 0 : robot.base_num_dof] = 0.0
            #actions[:, -1] = 1
            #print(">>>>>>>> Reset! Opening gripper.")
            #print(ISAAC_ORBIT_NUCLEUS_DIR)
        # change the gripper action
        if ep_step_count % 200 == 0:
            # flip command
            pass
            #actions[:, -1] = -actions[:, -1]
        # apply action
        #robot.apply_action(actions)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)
            # read buffers
            # if ep_step_count % 20 == 0:
            #     if robot.data.tool_dof_pos[0, -1] > 0.01:
            #         print("Opened gripper.")
            #     else:
            #         print("Closed gripper.")

# from omni.physx.scripts.physicsUtils import *
# def on_contact_report_event(contact_headers, contact_data):
#     for contact_header in contact_headers:
#         print("Got contact header type: " + str(contact_header.type))
#         print("Actor0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0)))
#         print("Actor1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1)))
#         print("Collider0: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider0)))
#         print("Collider1: " + str(PhysicsSchemaTools.intToSdfPath(contact_header.collider1)))
#         print("StageId: " + str(contact_header.stage_id))
#         print("Number of contacts: " + str(contact_header.num_contact_data))
        
#         contact_data_offset = contact_header.contact_data_offset

if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
